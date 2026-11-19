# Diagnosi dei batch saltati (loss non finita) del Caso 0 su CIC-ToN-IoT.
# Solo CPU, nessun addestramento: ricostruisce grafi e fold come nel notebook
# baseline-paper-marco e prova ogni grafo da solo con il modello appena inizializzato.
# Uso sul server (dalla cartella TESI_GraphIDS):  python diagnosi_ton.py
# Uso su Kaggle (Internet attivo, CPU):  !python diagnosi_ton.py   -> scarica ToN da solo, come il notebook
import os, json, hashlib, zipfile, urllib.request
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # nessuna GPU
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "8"

import random
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tqdm import tqdm

torch.set_num_threads(8)

# ---- stessa configurazione del notebook ----
SU_KAGGLE = Path("/kaggle/working").is_dir()
BASE = Path("/kaggle/working") if SU_KAGGLE else Path(".")
PERCORSO = BASE / "data/cictoniot/CIC-ToN-IoT-graph.parquet"
OUT = BASE / "data/diagnosi_ton"
CSV_CHUNK_ROWS = 250_000
TONIOT_DOWNLOAD_API = "https://api.rdm.uq.edu.au/datasets/127784c0-ef9d-11ed-a964-b70596e96ad5/download"
TONIOT_CSV_SHA1 = "5a1e5954512212b4511adbd60ba509bd5d19fc5c"
TONIOT_CSV_BYTES = 2_989_504_284
SEED, SPLIT_SEED, N_FOLDS, KFOLD_VAL_FRACTION = 123, 123, 5, 0.15
WINDOW_MINUTES = 1
FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0
PARAMS = {"hidden_channels": 64, "out_channels": 32, "dropout": 0.2}
SEMANTICS = {
    "network": ["Source Port", "Destination Port"],
    "context": ["Flow Duration", "Flow IAT Mean"],
    "knowledge": ["Packet Length Std", "SYN Flag Count"],
}
FEATURES = [f for g in SEMANTICS.values() for f in g]
REQUIRED = ["Source IP", "Destination IP", "Timestamp", "Label", *FEATURES]
RELATIONS = [("ip", name, "ip") for name in SEMANTICS]
ALIASES = {"Src IP": "Source IP", "Dst IP": "Destination IP", "Src Port": "Source Port",
           "Dst Port": "Destination Port", "Pkt Len Std": "Packet Length Std", "SYN Flag Cnt": "SYN Flag Count"}
SALTATI_REALI = {1: 3, 2: 2, 3: 0, 4: 2, 5: 1}   # dal run 20261117_130036_041688 (per epoca)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---- download e conversione (copiati da prepare_toniot / convert_toniot_csv del notebook) ----
def write_data_stream(source, target, total=None, expected_hash=None, algorithm="sha256"):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    digest, written = hashlib.new(algorithm), 0
    try:
        with temporary.open("wb") as dest, tqdm(total=total, desc=target.name, unit="B", unit_scale=True) as bar:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(chunk); digest.update(chunk); bar.update(len(chunk)); written += len(chunk)
        if total is not None and written != total:
            raise ValueError(f"Download incompleto: {target.name}")
        if expected_hash is not None and digest.hexdigest() != expected_hash:
            raise ValueError(f"Checksum non valido: {target.name}")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def convert_toniot_csv(source, target):
    temporary = target.with_name(target.name + ".part")
    columns = pd.read_csv(source, nrows=0, encoding="latin1").columns
    rename = {c: ALIASES.get(c.strip(), c.strip()) for c in columns}
    usecols = [c for c in columns if rename[c] in [*REQUIRED, "Attack"]]
    writer = None
    try:
        with pd.read_csv(source, usecols=usecols, chunksize=CSV_CHUNK_ROWS, encoding="latin1") as reader, \
                tqdm(desc="Parquet CIC-ToN-IoT", unit="flow") as bar:
            for chunk in reader:
                chunk = chunk.rename(columns=rename)
                for c in ["Source IP", "Destination IP", "Label", "Attack"]:
                    if c in chunk:
                        chunk[c] = chunk[c].astype("string")
                chunk[FEATURES] = chunk[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64")
                chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"].astype("string").str.upper(),
                                                    format="%d/%m/%Y %I:%M:%S %p", errors="raise")
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
                writer.write_table(table); bar.update(len(chunk))
        writer.close(); writer = None
        temporary.replace(target)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def prepare_toniot(target):
    if target.is_file():
        return target
    csv = target.parent / "original" / "CIC-ToN-IoT.csv"
    if not csv.is_file():
        archive = target.parent / "original" / "CIC-ToN-IoT.zip"
        if not archive.is_file():
            request = urllib.request.Request(TONIOT_DOWNLOAD_API, headers={"User-Agent": "GraphIDS-Marco/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                url = json.load(response).get("url", "")
            request = urllib.request.Request(url, headers={"User-Agent": "GraphIDS-Marco/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                write_data_stream(response, archive, int(length) if length else None)
        with zipfile.ZipFile(archive) as zipped:
            members = [i for i in zipped.infolist() if Path(i.filename).name == "CIC-ToN-IoT.csv"]
            if len(members) != 1 or members[0].file_size != TONIOT_CSV_BYTES:
                raise ValueError("Archivio UQ inatteso")
            with zipped.open(members[0]) as source:
                write_data_stream(source, csv, members[0].file_size, TONIOT_CSV_SHA1, "sha1")
        archive.unlink()      # libera spazio (su Kaggle il disco è limitato)
    convert_toniot_csv(csv, target)
    csv.unlink()
    return target


# ---- dati (copiati da load_flows / load_dataset / build_graph) ----
def load_flows(path):
    columns = pq.ParquetFile(path).schema.names
    rename = {c: ALIASES.get(c.strip(), c.strip()) for c in columns}
    usecols = [c for c in columns if rename[c] in [*REQUIRED, "Attack"]]
    raw = pd.read_parquet(path, columns=usecols).rename(columns=rename)
    rows = len(raw)
    for c in ["Source IP", "Destination IP", "Label"]:
        raw[c] = raw[c].astype("string").str.strip().replace("", pd.NA)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], format="mixed", dayfirst=True, errors="coerce")
    raw[FEATURES] = raw[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna(subset=REQUIRED).copy()
    numeric = pd.to_numeric(raw["Label"], errors="coerce")
    if numeric.notna().any():
        raw["target"] = numeric.astype("int64")
    else:
        raw["target"] = (~raw["Label"].str.upper().isin(["BENIGN", "NORMAL"])).astype("int64")
    raw["attack_type"] = raw["Attack"].astype("string").fillna(raw["Label"]) if "Attack" in raw else raw["Label"]
    raw.loc[raw["target"] == 0, "attack_type"] = "BENIGN"
    df = raw[[*REQUIRED, "target", "attack_type"]].sort_values("Timestamp", kind="stable").reset_index(drop=True)
    df["flow_id"] = np.arange(len(df))
    df["window"] = df["Timestamp"].dt.floor(f"{WINDOW_MINUTES}min")
    print(f"Letti {rows:,}; flow usati {len(df):,}; finestre {df.window.nunique():,}")
    return df


def build_graph(window_df):
    graph = nx.MultiDiGraph()
    graph.add_edges_from(zip(window_df["Source IP"], window_df["Destination IP"]))
    nodes = list(graph.nodes)
    mapping = {node: i for i, node in enumerate(nodes)}
    communities = nx.community.label_propagation_communities(nx.Graph(graph))
    community_id = {node: c for c, members in enumerate(communities) for node in members}
    data = HeteroData()
    data["ip"].num_nodes = len(nodes)
    data["ip"].x = torch.tensor([[community_id[n], graph.in_degree(n), graph.out_degree(n)] for n in nodes],
                                dtype=torch.float32)
    edge_index = torch.tensor([window_df["Source IP"].map(mapping).tolist(),
                               window_df["Destination IP"].map(mapping).tolist()], dtype=torch.long)
    for relation, features in SEMANTICS.items():
        store = data["ip", relation, "ip"]
        store.edge_index = edge_index.clone()
        store.edge_attr = torch.tensor(window_df[features].to_numpy(dtype=np.float32))
        store.edge_label = torch.tensor(window_df["target"].to_numpy(), dtype=torch.long)
    data.window = str(window_df["window"].iloc[0])
    return data


# ---- fold (copiati da evaluation_splits) ----
def evaluation_splits(graphs):
    strata = np.array([int(g[RELATIONS[0]].edge_label.max()) for g in graphs])
    counts = np.unique(strata, return_counts=True)[1]
    splitter = (StratifiedKFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED) if min(counts) >= N_FOLDS
                else KFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED))
    folds = {}
    for fold, (remaining, test) in enumerate(splitter.split(np.arange(len(graphs)), strata), 1):
        labels = strata[remaining]
        classes, freq = np.unique(labels, return_counts=True)
        val_size = int(np.ceil(KFOLD_VAL_FRACTION * len(remaining)))
        stratify = labels if min(freq) >= 2 and val_size >= len(classes) and len(remaining) - val_size >= len(classes) else None
        train, val = train_test_split(remaining, test_size=KFOLD_VAL_FRACTION, random_state=SPLIT_SEED, stratify=stratify)
        folds[fold] = {"train": train, "validation": val, "test": test}
    return folds


# ---- modello Caso 0 (copiato dal notebook) ----
class HeteroEdgeClassifier(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=32, num_classes=2, dropout=0.2):
        super().__init__()
        self.conv1 = HeteroConv({r: GCNConv(-1, hidden_channels) for r in RELATIONS}, aggr="sum")
        self.bn1_ip = nn.BatchNorm1d(hidden_channels)
        self.conv2 = HeteroConv({r: GCNConv(hidden_channels, hidden_channels) for r in RELATIONS}, aggr="sum")
        self.bn2_ip = nn.BatchNorm1d(hidden_channels)
        self.lin_node = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 2), nn.ReLU(), nn.Dropout(p=dropout),
                                      nn.Linear(hidden_channels * 2, out_channels), nn.ReLU())
        self.lin_edge = nn.Sequential(nn.Linear(out_channels * 2, out_channels), nn.ReLU(), nn.Dropout(p=dropout),
                                      nn.Linear(out_channels, out_channels), nn.ReLU(), nn.Dropout(p=dropout),
                                      nn.Linear(out_channels, num_classes))

    def edge_classify(self, z_src, z_dst):
        return self.lin_edge(torch.cat([z_src, z_dst], dim=-1))


def phi_values(edge_attr):
    return torch.log1p(edge_attr.abs())


def fit_phi_stats(train_graphs):
    stats = {}
    for relation in RELATIONS:
        values = torch.cat([phi_values(g[relation].edge_attr) for g in train_graphs])
        stats[relation[1]] = (values.min(dim=0).values, values.max(dim=0).values)
    return stats


class PaperEdgeClassifier(HeteroEdgeClassifier):
    def __init__(self, **kw):
        super().__init__(**kw)
        init = float(np.log(np.expm1(0.5)))
        self.phi_alpha = nn.ParameterDict({r[1]: nn.Parameter(torch.full((2,), init)) for r in RELATIONS})
        for r in RELATIONS:
            self.register_buffer(f"phi_min_{r[1]}", torch.zeros(2))
            self.register_buffer(f"phi_max_{r[1]}", torch.ones(2))

    def set_phi_stats(self, stats):
        for name, (low, high) in stats.items():
            getattr(self, f"phi_min_{name}").copy_(low)
            getattr(self, f"phi_max_{name}").copy_(high)

    def edge_weights(self, data):
        weights = {}
        for relation in RELATIONS:
            name = relation[1]
            low, high = getattr(self, f"phi_min_{name}"), getattr(self, f"phi_max_{name}")
            norm = ((phi_values(data[relation].edge_attr) - low) / (high - low).clamp_min(1e-12)).clamp(0.0, 1.0)
            weights[relation] = (norm * F.softplus(self.phi_alpha[name])).sum(dim=-1)
        return weights

    def _bn(self, bn, x):
        return (bn(x) if x.size(0) > 1 or not self.training else
                F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, training=False))

    def forward(self, data, trace=None):
        def check(name, t):
            if trace is not None and not trace and not torch.isfinite(t).all():
                trace.append(name)
            return t
        ew = self.edge_weights(data)
        for r, w in ew.items():
            check(f"phi_{r[1]}", w)
        x = self.conv1(data.x_dict, data.edge_index_dict, edge_weight_dict=ew)["ip"]
        x = F.relu(check("bn1", self._bn(self.bn1_ip, check("conv1", x))))
        x = self.conv2({"ip": x}, data.edge_index_dict, edge_weight_dict=ew)["ip"]
        x = F.relu(check("bn2", self._bn(self.bn2_ip, check("conv2", x))))
        return check("lin_node", self.lin_node(x))


def focal_terms(logits, targets):
    ce = F.cross_entropy(logits, targets, reduction="none")
    return FOCAL_ALPHA * (1 - torch.exp(-ce)) ** FOCAL_GAMMA * ce


@torch.no_grad()
def prova_grafo(model, g):
    trace = []
    z = model(g, trace)
    loss, max_logit, max_term = 0.0, 0.0, 0.0
    for r in RELATIONS:
        logits = model.edge_classify(z[g[r].edge_index[0]], z[g[r].edge_index[1]])
        if not trace and not torch.isfinite(logits).all():
            trace.append("logits")
        terms = focal_terms(logits, g[r].edge_label)
        if not trace and not torch.isfinite(terms).all():
            trace.append("focal_per_arco")
        loss = loss + terms.mean()
        max_logit = max(max_logit, float(logits.abs().max()))
        max_term = max(max_term, float(terms.max()))
    finita = bool(torch.isfinite(torch.as_tensor(loss)))
    if not finita and not trace:
        trace.append("media_loss (overflow della somma)")
    return finita, (trace[0] if trace else ""), float(loss), max_logit, max_term


def statistiche_grafo(g):
    x = g["ip"].x
    ea = torch.cat([g[r].edge_attr for r in RELATIONS], dim=1)  # 6 feature, stesso ordine di FEATURES
    src, dst = g[RELATIONS[0]].edge_index
    row = {"window": g.window, "nodi": int(x.size(0)), "archi": int(src.numel()),
           "attacchi": int(g[RELATIONS[0]].edge_label.sum()),
           "max_in_degree": float(x[:, 1].max()), "max_out_degree": float(x[:, 2].max()),
           "max_community": float(x[:, 0].max()), "self_loop": int((src == dst).sum()),
           "feature_non_finite_float32": int((~torch.isfinite(ea)).sum())}
    for i, f in enumerate(FEATURES):
        row[f"max|{f}|"] = float(ea[:, i].abs().max())
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_flows(prepare_toniot(PERCORSO))
    famiglie = df.groupby("window")["attack_type"].agg(lambda s: ",".join(sorted(set(s) - {"BENIGN"})))
    groups = list(df.groupby("window", sort=True))
    graphs = [build_graph(part) for _, part in tqdm(groups, desc="Grafi ToN")]
    del groups
    assert sum(g[RELATIONS[0]].num_edges for g in graphs) == len(df)

    tabella = pd.DataFrame([statistiche_grafo(g) for g in graphs])
    tabella["famiglie"] = famiglie.reindex(pd.to_datetime(tabella["window"])).values
    folds = evaluation_splits(graphs)
    for f, parts in folds.items():
        col = np.empty(len(graphs), dtype=object)
        for name, idx in parts.items():
            col[idx] = name
        tabella[f"fold{f}"] = col

    print("\nFeature non finite dopo il cast a float32:", int(tabella.feature_non_finite_float32.sum()))

    riepilogo = []
    for f, parts in folds.items():
        stats = fit_phi_stats([graphs[i] for i in parts["train"]])
        print(f"\nFold {f} | min-max Φ:", {k: [v.tolist() for v in p] for k, p in stats.items()})
        seed_everything(SEED)
        model = PaperEdgeClassifier(**PARAMS)
        model.set_phi_stats(stats)
        with torch.no_grad():
            model.train(); model(graphs[parts["train"][0]])   # inizializza GCNConv(-1, ...)
        res = {}
        for mode in ["train", "eval"]:
            model.train(mode == "train")
            for i in tqdm(range(len(graphs)), desc=f"fold {f} ({mode})", leave=False):
                res.setdefault(i, {}).update(dict(zip(
                    [f"f{f}_{mode}_finita", f"f{f}_{mode}_dove", f"f{f}_{mode}_loss", f"f{f}_{mode}_max_logit", f"f{f}_{mode}_max_termine"],
                    prova_grafo(model, graphs[i]))))
        res = pd.DataFrame.from_dict(res, orient="index").sort_index()
        tabella = pd.concat([tabella, res], axis=1)
        cattivi = ~tabella[f"f{f}_train_finita"]
        nel_train = int((cattivi & (tabella[f"fold{f}"] == "train")).sum())
        riepilogo.append({"fold": f, "grafi non finiti (train mode, tutti)": int(cattivi.sum()),
                          "di cui nel train": nel_train, "batch saltati nel run vero": SALTATI_REALI[f],
                          "non finiti in eval mode": int((~tabella[f"f{f}_eval_finita"]).sum())})

    tabella.to_csv(OUT / "diagnosi_grafi.csv", index=False)
    riepilogo = pd.DataFrame(riepilogo)
    riepilogo.to_csv(OUT / "riepilogo.csv", index=False)
    print("\n=== RIEPILOGO ===")
    print(riepilogo.to_string(index=False))

    colonne_cattivi = [c for c in tabella.columns if c.endswith("_train_finita")]
    sospetti = tabella[~tabella[colonne_cattivi].all(axis=1)]
    base = ["window", "nodi", "archi", "attacchi", "famiglie", "max_in_degree", "max_out_degree",
            "self_loop", "feature_non_finite_float32", *[f"fold{f}" for f in folds],
            *[f"f{f}_train_dove" for f in folds]]
    print("\n=== GRAFI CON LOSS NON FINITA ===")
    print(sospetti[base].to_string(index=False) if len(sospetti) else "nessuno")
    print("\nPer confronto, i 5 grafi più grandi:")
    print(tabella.nlargest(5, "archi")[base[:9]].to_string(index=False))
    sospetti.to_csv(OUT / "grafi_sospetti.csv", index=False)
    print(f"\nFile salvati in {OUT}/")


if __name__ == "__main__":
    main()
