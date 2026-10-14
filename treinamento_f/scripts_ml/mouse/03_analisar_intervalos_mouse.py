import hashlib
import json
from pathlib import Path

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

SNAPSHOT_DIR = BASE_DIR / "data" / "mouse_snapshot"

CSV_PATH = SNAPSHOT_DIR / "mouse_events_raw.csv"

METADATA_PATH = SNAPSHOT_DIR / "mouse_events_metadata.json"

OUTPUT_DIR = BASE_DIR / "data" / "mouse_analise"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. VERIFICAR SE O ARQUIVO ORIGINAL CONTINUA ÍNTEGRO
# ============================================================

print("=" * 75)
print("ANÁLISE DOS INTERVALOS E SESSÕES — MOUSE/TRACKPAD")
print("=" * 75)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado:\n{CSV_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadados não encontrados:\n{METADATA_PATH}"
    )


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as arquivo:

    metadata = json.load(arquivo)


hash_esperado = metadata.get("sha256")


print("\nVerificando SHA-256 do snapshot...")

sha256 = hashlib.sha256()

with open(
    CSV_PATH,
    "rb"
) as arquivo:

    while True:

        bloco = arquivo.read(
            1024 * 1024
        )

        if not bloco:
            break

        sha256.update(bloco)


hash_atual = sha256.hexdigest()


print(f"Hash esperado: {hash_esperado}")
print(f"Hash atual:    {hash_atual}")


if hash_atual != hash_esperado:

    raise RuntimeError(
        "\nATENÇÃO: o arquivo raw foi alterado.\n"
        "A análise foi interrompida."
    )


print("✅ Snapshot íntegro.")


# ============================================================
# 2. CARREGAR O CSV
# ============================================================

print("\nCarregando os dados...")

df = pd.read_csv(
    CSV_PATH,
    low_memory=False
)

print(
    f"✅ {len(df)} registros carregados."
)


# ============================================================
# 3. VALIDAÇÕES BÁSICAS
# ============================================================

print("\n" + "=" * 75)
print("VALIDAÇÃO BÁSICA")
print("=" * 75)


print(
    f"\nRegistros: {len(df)}"
)

print(
    f"Usuárias: {df['user'].nunique()}"
)

print(
    f"IDs duplicados: {df['_id'].duplicated().sum()}"
)


campos_criticos = [
    "_id",
    "user",
    "timestamp",
    "event_type",
    "device_type"
]


print("\nValores ausentes:")

for campo in campos_criticos:

    faltantes = df[campo].isna().sum()

    print(
        f"  {campo:<15}: {faltantes}"
    )


# ============================================================
# 4. CONVERTER TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_numeric(
    df["timestamp"],
    errors="coerce"
)


invalidos = df["timestamp"].isna().sum()

print(
    f"\nTimestamps inválidos: {invalidos}"
)


df = df.dropna(
    subset=["timestamp"]
).copy()


df["datetime_br"] = (
    pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )
    .dt.tz_convert(
        "America/Sao_Paulo"
    )
)


# ============================================================
# 5. ORDENAR LOCALMENTE
# ============================================================

df = df.sort_values(
    ["user", "timestamp"]
).reset_index(drop=True)


# ============================================================
# 6. CALCULAR INTERVALO ENTRE EVENTOS
# ============================================================

df["gap_segundos"] = (
    df
    .groupby("user")["timestamp"]
    .diff()
    / 1000
)


# ============================================================
# 7. RESUMO DOS INTERVALOS
# ============================================================

print("\n" + "=" * 75)
print("INTERVALOS ENTRE EVENTOS POR USUÁRIA")
print("=" * 75)


resumos = []


for usuario, grupo in df.groupby("user"):

    gaps = grupo["gap_segundos"].dropna()

    primeiro = grupo["datetime_br"].iloc[0]

    ultimo = grupo["datetime_br"].iloc[-1]

    resumo = {

        "user": usuario,

        "eventos": len(grupo),

        "inicio": str(primeiro),

        "fim": str(ultimo),

        "gap_mediana_s": gaps.median(),

        "gap_p90_s": gaps.quantile(0.90),

        "gap_p95_s": gaps.quantile(0.95),

        "gap_p99_s": gaps.quantile(0.99),

        "gap_max_s": gaps.max(),

        "gaps_maior_30s": (
            gaps > 30
        ).sum(),

        "gaps_maior_60s": (
            gaps > 60
        ).sum(),

        "gaps_maior_120s": (
            gaps > 120
        ).sum(),

        "gaps_maior_300s": (
            gaps > 300
        ).sum()
    }

    resumos.append(resumo)

    print(
        f"\nUsuária: {usuario}"
    )

    print(
        f"  Eventos: {len(grupo)}"
    )

    print(
        f"  Mediana do intervalo: "
        f"{gaps.median():.3f} s"
    )

    print(
        f"  Percentil 90: "
        f"{gaps.quantile(0.90):.3f} s"
    )

    print(
        f"  Percentil 95: "
        f"{gaps.quantile(0.95):.3f} s"
    )

    print(
        f"  Percentil 99: "
        f"{gaps.quantile(0.99):.3f} s"
    )

    print(
        f"  Maior intervalo: "
        f"{gaps.max():.2f} s"
    )

    print(
        f"  Pausas > 30 s:  "
        f"{(gaps > 30).sum()}"
    )

    print(
        f"  Pausas > 60 s:  "
        f"{(gaps > 60).sum()}"
    )

    print(
        f"  Pausas > 120 s: "
        f"{(gaps > 120).sum()}"
    )

    print(
        f"  Pausas > 300 s: "
        f"{(gaps > 300).sum()}"
    )


resumo_df = pd.DataFrame(
    resumos
)


resumo_path = (
    OUTPUT_DIR
    / "resumo_intervalos_por_usuario.csv"
)


resumo_df.to_csv(
    resumo_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. COMPARAR POSSÍVEIS LIMITES DE SESSÃO
# ============================================================

print("\n" + "=" * 75)
print("QUANTIDADE DE SESSÕES PARA DIFERENTES LIMITES")
print("=" * 75)


limiares = [
    30,
    60,
    120,
    300,
    600
]


resultados_sessoes = []


for usuario, grupo in df.groupby("user"):

    print(
        f"\n{usuario}"
    )

    for limite in limiares:

        novas_sessoes = (
            grupo["gap_segundos"].isna()
            |
            (
                grupo["gap_segundos"]
                > limite
            )
        )

        quantidade = int(
            novas_sessoes.sum()
        )

        resultados_sessoes.append({

            "user": usuario,

            "limite_segundos": limite,

            "quantidade_sessoes": quantidade
        })

        print(
            f"  pausa > {limite:>3}s "
            f"→ {quantidade} sessões"
        )


pd.DataFrame(
    resultados_sessoes
).to_csv(

    OUTPUT_DIR
    / "sessoes_por_limiar.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 9. IDENTIFICAR MUDANÇAS MOUSE <-> TRACKPAD
# ============================================================

print("\n" + "=" * 75)
print("TRANSIÇÕES DE DEVICE_TYPE")
print("=" * 75)


df["device_anterior"] = (
    df
    .groupby("user")["device_type"]
    .shift(1)
)


mudancas = df[
    df["device_anterior"].notna()
    &
    (
        df["device_type"]
        != df["device_anterior"]
    )
].copy()


print(
    f"\nTotal de mudanças de rótulo: "
    f"{len(mudancas)}"
)


for usuario in sorted(
    df["user"].dropna().unique()
):

    quantidade = len(
        mudancas[
            mudancas["user"]
            == usuario
        ]
    )

    print(
        f"  {usuario:<22}: "
        f"{quantidade}"
    )


transicoes_path = (
    OUTPUT_DIR
    / "transicoes_device_type.csv"
)


mudancas[
    [
        "user",
        "datetime_br",
        "event_type",
        "device_anterior",
        "device_type",
        "gap_segundos"
    ]
].to_csv(
    transicoes_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. MOSTRAR TRANSIÇÕES DA BIANCA
# ============================================================

bianca_transicoes = mudancas[
    mudancas["user"]
    .astype(str)
    .str.lower()
    == "bianca"
].copy()


print("\n" + "=" * 75)
print("PRIMEIRAS MUDANÇAS DE DISPOSITIVO — BIANCA")
print("=" * 75)


if len(bianca_transicoes) == 0:

    print(
        "\nNenhuma mudança encontrada."
    )

else:

    for _, linha in (
        bianca_transicoes
        .head(30)
        .iterrows()
    ):

        print(
            f"{linha['datetime_br']} | "
            f"{linha['device_anterior']} "
            f"→ {linha['device_type']} | "
            f"gap={linha['gap_segundos']:.3f}s"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("ANÁLISE CONCLUÍDA")
print("=" * 75)

print(
    f"\nResumo salvo em:\n{resumo_path}"
)

print(
    f"\nTransições salvas em:\n{transicoes_path}"
)

print(
    "\n✅ O arquivo RAW não foi modificado."
)