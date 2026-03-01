



# ==========================================
# Build Neighborhood Schema (EnergyPlus + CityLearn)
# ==========================================
def build_neighborhood_schema(idd_filepath: str, sample_count: int = 2, filters: dict = None, delete_energyplus_simulation_output: bool = True):
    from citylearn.end_use_load_profiles.neighborhood import Neighborhood, SampleMethod
    if filters is None:
        filters = {
            "in.resstock_county_id": ["TX, Travis County"],
            "in.vintage": ["2000s"],
        }
    build_kwargs = dict(
        idd_filepath=idd_filepath,
        delete_energyplus_simulation_output=delete_energyplus_simulation_output,
        sample_buildings_kwargs=dict(
            sample_method=SampleMethod.RANDOM,
            sample_count=sample_count,
            filters=filters,
        ),
    )
    neighborhood = Neighborhood()
    nb = neighborhood.build(**build_kwargs)
    schema_path = nb.schema_filepath
    print(f"[Neighborhood] schema generated at: {schema_path}")
    return schema_path


# ==========================================
# Create CityLearn Environment
# ==========================================
def create_citylearn_env(
    dataset: str = "citylearn_challenge_2023_phase_1",
    central: bool = True,
    use_neighborhood: bool = False,
    idd_filepath: str = None,
    neighborhood_schema_path: str = None,
    neighborhood_build_kwargs: dict = None,
    optimize_grid: bool = True,
    node_building_number: int = None,
):

    if use_neighborhood:
        if neighborhood_schema_path is None:
            if idd_filepath is None:
                raise ValueError(
                    "When use_neighborhood=True, either neighborhood_schema_path "
                    "or idd_filepath must be provided."
                )
            schema_path = build_neighborhood_schema(
                idd_filepath=idd_filepath,
                **(neighborhood_build_kwargs or {}),
            )
        else:
            schema_path = neighborhood_schema_path

        data = schema_path

    else:
        data = dataset

    env_kwargs = dict(
        central_agent=central,
        reward_function=VoltageReward if optimize_grid else RewardFunction,
    )

    if node_building_number is not None:
        env_kwargs["node_building_number"] = node_building_number

    env = CityLearnEnv(data, **env_kwargs)

    return env


# ==========================================
# Create CityLearn Agent
# ==========================================
def create_citylearn_agent(env, strategy: str, episodes: int):
    if strategy == "RBC":
        agent = Agent_RBC(env)
    elif strategy == "SAC":
        agent = Agent_SAC(env)
        agent.learn(episodes=episodes, deterministic_finish=True)
    elif strategy == "BASELINE":
        agent = Agent_Baseline(env)
    else:
        raise ValueError(f"Unsupported CityLearn strategy: {strategy}")
    return agent


# ==========================================
# Evaluate CityLearn KPIs
# ==========================================
def evaluate_citylearn_kpis(agent):
    kpis = agent.env.evaluate()
    kpis = kpis.pivot(index="cost_function", columns="name", values="value").round(3)
    kpis = kpis.dropna(how="all")
    print(kpis)
    return kpis


# ==========================================
# Run Full CityLearn Simulation
# ==========================================
def run_citylearn(
    strategy="RBC",
    episodes=2,
    central=True,
    dataset="citylearn_challenge_2023_phase_1",
    use_neighborhood=True,
    idd_filepath=r'C:\EnergyPlusV9-6-0\PreProcess\IDFVersionUpdater\V9-6-0-Energy+.idd',
    neighborhood_schema_path=None,
    neighborhood_build_kwargs=None,
    optimize_grid=False,
    node_building_number=None,
):
    env = create_citylearn_env(
        dataset=dataset,
        central=central,
        use_neighborhood=use_neighborhood,
        idd_filepath=idd_filepath,
        neighborhood_schema_path=neighborhood_schema_path,
        neighborhood_build_kwargs=neighborhood_build_kwargs,
        optimize_grid=optimize_grid,
        node_building_number=node_building_number,
    )
    model = create_citylearn_agent(
        env=env,
        strategy=strategy,
        episodes=episodes,
    )
    observations, _ = env.reset()
    env.unwrapped.bus_voltages_history = []

    while not env.terminated:
        actions = model.predict(observations, deterministic=True)
        observations, reward, info, terminated, truncated = env.step(actions)

    buildings = env.unwrapped.buildings
    building_kw = np.stack(
        [b.net_electricity_consumption for b in buildings],
        axis=1
    )
    if node_building_number is not None:
        building_kw = building_kw * (node_building_number / len(buildings))
    print(f"[CityLearn] building_kw shape = {building_kw.shape}")
    return building_kw, model


# ==========================================
# Load Power Network Case (Pandapower)
# ==========================================
def load_network(case_name="case33bw"):
    case_name = case_name.lower()

    case_map = {
        "case9": pn.case9,
        "case14": pn.case14,
        "case30": pn.case30,
        "case33bw": pn.case33bw,
        "case57": pn.case57,
        "case118": pn.case118,
    }

    if case_name not in case_map:
        supported = ", ".join(case_map.keys())
        raise ValueError(
            f"Unsupported case '{case_name}'. Supported cases: {supported}"
        )

    net = case_map[case_name]()
    net.line["max_i_ka"] = 0.5
    net.load.drop(net.load.index, inplace=True)
    pp.create_shunt(net, bus=14, q_mvar=-1.2, p_mw=0.0)

    print(f"[NetInit] {case_name}: buses={len(net.bus)}, lines={len(net.line)}")
    return net


# ==========================================
# Run Time-Series Power Flow Simulation
# ==========================================
def run_grid(building_kw, net):

    T, _ = building_kw.shape
    n_buses = len(net.bus)

    total_mw = building_kw.sum(axis=1) / 1000.0

    net_ts = copy.deepcopy(net)

    load_idx = {}
    for bus in range(1, n_buses):
        load_idx[bus] = pp.create_load(net_ts, bus=bus, p_mw=0.0)

    vm_ts = []
    loading_ts = []

    for t in range(T):
        for bus, idx in load_idx.items():
            net_ts.load.at[idx, "p_mw"] = total_mw[t]
        pp.runpp(net_ts)
        vm_ts.append(net_ts.res_bus.vm_pu.values.copy())
        loading_ts.append(net_ts.res_line.loading_percent.values.copy())

    grid_results = {
        "total_mw": total_mw,
        "vm_ts": np.array(vm_ts),
        "loading_ts": np.array(loading_ts),
        "net": net,
        "net_ts": net_ts,
    }

    print("[Grid] powerflow finished")
    return grid_results


# ==========================================
# Plot Grid Simulation Results
# ==========================================
def plot_grid_results(res: dict):

    vm_ts = res["vm_ts"]
    loading_ts = res["loading_ts"]

    volt_df = pd.DataFrame(vm_ts)
    plt.figure()
    volt_df.plot(legend=False)
    plt.xlabel("Time step")
    plt.ylabel("Voltage [p.u.]")
    plt.title("Bus voltages over time")
    plt.tight_layout()
    plt.savefig(picture_path_voltages)
    plt.close()

    line_df = pd.DataFrame(loading_ts)
    plt.figure()
    line_df.plot(legend=False)
    plt.xlabel("Time step")
    plt.ylabel("Line loading [%]")
    plt.title("Line loadings over time")
    plt.tight_layout()
    plt.savefig(picture_path_lines)
    plt.close()

    print("[Grid] figures saved")


# ==========================================
# Save Grid Simulation Results to CSV
# ==========================================
def save_grid_results(res: dict):

    pd.DataFrame(res["vm_ts"]).to_csv(
        os.path.join(DATA_DIR, "voltages_ts.csv"), index=False
    )

    pd.DataFrame(res["loading_ts"]).to_csv(
        os.path.join(DATA_DIR, "line_loading_ts.csv"), index=False
    )

    pd.DataFrame({"total_mw": res["total_mw"]}).to_csv(
        os.path.join(DATA_DIR, "system_total_load.csv"), index=False
    )

    print("[Grid] CSV files saved")


# ==========================================
# Perform N-1 Contingency Analysis
# ==========================================
def analyze_n1(net, building_kw,
               vm_tol=0.05,
               loading_threshold=70.0,
               T_run=24):

    print("[N-1] analysis started")

    net_ts = copy.deepcopy(net)
    T_total, _ = building_kw.shape
    n_buses = len(net.bus)

    T_use = T_total if T_run is None else min(int(T_run), T_total)

    load_idx = {}
    for bus in range(1, n_buses):
        load_idx[bus] = pp.create_load(net_ts, bus=bus, p_mw=0.0)

    total_mw = building_kw.sum(axis=1) / 1000.0

    under_ts, over_ts, overload_ts = [], [], []

    for t in range(T_use):
        for bus, idx in load_idx.items():
            net_ts.load.at[idx, "p_mw"] = total_mw[t]

        pp.runpp(net_ts)

        total_under = 0
        total_over = 0
        total_overload = 0

        for line_idx in net_ts.line.index:
            net_cont = copy.deepcopy(net_ts)

            net_cont.line.at[line_idx, "in_service"] = False

            try:
                pp.runpp(net_cont)
            except Exception:
                total_under += 1
                continue

            if not net_cont.converged:
                total_under += 1
                continue

            vm = net_cont.res_bus.vm_pu.values
            loading = net_cont.res_line.loading_percent.values

            total_under += int((vm < 1.0 - vm_tol).sum())
            total_over += int((vm > 1.0 + vm_tol).sum())
            total_overload += int((loading > loading_threshold).sum())

        under_ts.append(total_under)
        over_ts.append(total_over)
        overload_ts.append(total_overload)

    plt.figure()
    plt.plot(under_ts, label="undervoltage")
    plt.plot(over_ts, label="overvoltage")
    plt.plot(overload_ts, label="overload")
    plt.legend()
    plt.tight_layout()
    plt.savefig(picture_path_n1)
    plt.close()

    print(
        f"[N-1] analysis finished: "
        f"steps={T_use}, "
        f"max_under={np.max(under_ts):.0f}, "
        f"max_over={np.max(over_ts):.0f}, "
        f"max_overload={np.max(overload_ts):.0f}"
    )