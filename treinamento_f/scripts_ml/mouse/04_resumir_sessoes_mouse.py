import hashlib
import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

LIMITE_SESSAO_SEGUNDOS = 300

BASE_DIR = Path(__file__).resolve().parents[1]

SNAPSHOT_DIR = BASE_DIR / "data" / "mouse_snapshot"
OUTPUT_DIR = BASE_DIR / "data" / "mouse_analise"

CSV_PATH = SNAPSHOT_DIR / "mouse_events_raw.csv"
METADATA_PATH = SNAPSHOT_DIR / "mouse_events_metadata.json"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "resumo_sessoes_300s.csv"
)


# ============================================================
# 1. VERIFICAR INTEGRIDADE DO SNAPSHOT
# ============================================================

print("=" * 80)
print("RECONSTRUÇÃO DAS SESSÕES DE MOUSE/TRACKPAD")
print("=" * 80)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as arquivo:

    metadata = json.load(arquivo)


hash_esperado = metadata["sha256"]

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


if hash_atual != hash_esperado:

    raise RuntimeError(
        "O arquivo RAW foi alterado. "
        "Análise interrompida."
    )


print("\n✅ Snapshot íntegro.")


# ============================================================
# 2. CARREGAR OS DADOS
# ============================================================

print("\nCarregando dados...")

df = pd.read_csv(
    CSV_PATH,
    low_memory=False
)

df["timestamp"] = pd.to_numeric(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "user",
        "timestamp",
        "event_type",
        "device_type"
    ]
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
# 3. ORDENAR CRONOLOGICAMENTE
# ============================================================

df = df.sort_values(
    [
        "user",
        "timestamp"
    ]
).reset_index(drop=True)


# ============================================================
# 4. CALCULAR INTERVALOS
# ============================================================

df["gap_segundos"] = (
    df
    .groupby("user")["timestamp"]
    .diff()
    / 1000
)


# ============================================================
# 5. CRIAR AS SESSÕES
# ============================================================

nova_sessao = (
    df["gap_segundos"].isna()
    |
    (
        df["gap_segundos"]
        > LIMITE_SESSAO_SEGUNDOS
    )
)


df["session_number"] = (
    nova_sessao
    .groupby(df["user"])
    .cumsum()
    .astype(int)
)


df["session_id"] = (
    df["user"].astype(str)
    + "_S"
    + df["session_number"]
        .astype(str)
        .str.zfill(2)
)


# ============================================================
# 6. IDENTIFICAR TRANSIÇÕES DENTRO DA SESSÃO
# ============================================================

df["device_anterior_sessao"] = (
    df
    .groupby(
        [
            "user",
            "session_id"
        ]
    )["device_type"]
    .shift(1)
)


df["mudou_device"] = (
    df["device_anterior_sessao"].notna()
    &
    (
        df["device_type"]
        != df["device_anterior_sessao"]
    )
)


# ============================================================
# 7. RESUMIR CADA SESSÃO
# ============================================================

resultados = []


for (
    usuario,
    session_id
), grupo in df.groupby(
    [
        "user",
        "session_id"
    ],
    sort=False
):

    inicio = grupo["datetime_br"].iloc[0]

    fim = grupo["datetime_br"].iloc[-1]

    duracao_segundos = (
        grupo["timestamp"].iloc[-1]
        -
        grupo["timestamp"].iloc[0]
    ) / 1000


    total = len(grupo)


    mouse = (
        grupo["device_type"]
        .eq("mouse")
        .sum()
    )

    trackpad = (
        grupo["device_type"]
        .eq("trackpad")
        .sum()
    )


    pct_mouse = (
        mouse / total * 100
        if total
        else 0
    )

    pct_trackpad = (
        trackpad / total * 100
        if total
        else 0
    )


    if mouse > trackpad:

        device_dominante = "mouse"
        pct_dominante = pct_mouse

    elif trackpad > mouse:

        device_dominante = "trackpad"
        pct_dominante = pct_trackpad

    else:

        device_dominante = "empate"
        pct_dominante = 50


    if pct_dominante >= 95:

        consistencia = "muito_estavel"

    elif pct_dominante >= 80:

        consistencia = "predominante"

    elif pct_dominante >= 60:

        consistencia = "mista"

    else:

        consistencia = "muito_mista"


    resultados.append({

        "user": usuario,

        "session_id": session_id,

        "inicio": inicio.strftime(
            "%d/%m/%Y %H:%M:%S"
        ),

        "fim": fim.strftime(
            "%d/%m/%Y %H:%M:%S"
        ),

        "duracao_min": round(
            duracao_segundos / 60,
            2
        ),

        "total_eventos": total,

        "mouse_eventos": mouse,

        "trackpad_eventos": trackpad,

        "pct_mouse": round(
            pct_mouse,
            2
        ),

        "pct_trackpad": round(
            pct_trackpad,
            2
        ),

        "device_dominante_heuristica":
            device_dominante,

        "pct_dominante":
            round(
                pct_dominante,
                2
            ),

        "consistencia_heuristica":
            consistencia,

        "transicoes_device":
            int(
                grupo[
                    "mudou_device"
                ].sum()
            ),

        "mouse_move": int(
            grupo["event_type"]
            .eq("mouse_move")
            .sum()
        ),

        "mouse_click": int(
            grupo["event_type"]
            .eq("mouse_click")
            .sum()
        ),

        "mouse_down": int(
            grupo["event_type"]
            .eq("mouse_down")
            .sum()
        ),

        "mouse_up": int(
            grupo["event_type"]
            .eq("mouse_up")
            .sum()
        ),

        "mouse_wheel": int(
            grupo["event_type"]
            .eq("mouse_wheel")
            .sum()
        )
    })


resumo = pd.DataFrame(
    resultados
)


# ============================================================
# 8. SALVAR RESULTADOS
# ============================================================

resumo.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. RESUMO GERAL
# ============================================================

print("\n" + "=" * 80)
print("SESSÕES ENCONTRADAS")
print("=" * 80)


for usuario in sorted(
    resumo["user"].unique()
):

    quantidade = len(
        resumo[
            resumo["user"]
            == usuario
        ]
    )

    print(
        f"{usuario:<22}: "
        f"{quantidade} sessões"
    )


# ============================================================
# 10. MOSTRAR AS SESSÕES DA BIANCA
# ============================================================

bianca = resumo[
    resumo["user"]
    .astype(str)
    .str.lower()
    == "bianca"
].copy()


print("\n" + "=" * 80)
print("SESSÕES DA BIANCA")
print("=" * 80)


if bianca.empty:

    print(
        "\nNenhuma sessão da Bianca encontrada."
    )

else:

    for _, linha in bianca.iterrows():

        print(
            f"\n{linha['session_id']}"
        )

        print(
            f"  Início: "
            f"{linha['inicio']}"
        )

        print(
            f"  Fim:    "
            f"{linha['fim']}"
        )

        print(
            f"  Duração: "
            f"{linha['duracao_min']} min"
        )

        print(
            f"  Eventos: "
            f"{linha['total_eventos']}"
        )

        print(
            f"  Mouse: "
            f"{linha['mouse_eventos']} "
            f"({linha['pct_mouse']}%)"
        )

        print(
            f"  Trackpad: "
            f"{linha['trackpad_eventos']} "
            f"({linha['pct_trackpad']}%)"
        )

        print(
            f"  Rótulo dominante da heurística: "
            f"{linha['device_dominante_heuristica']}"
        )

        print(
            f"  Consistência: "
            f"{linha['consistencia_heuristica']}"
        )

        print(
            f"  Transições mouse/trackpad: "
            f"{linha['transicoes_device']}"
        )

        print(
            f"  Wheel: "
            f"{linha['mouse_wheel']}"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)

print("CONCLUÍDO")

print("=" * 80)

print(
    f"\nArquivo gerado:\n{OUTPUT_PATH}"
)

print(
    "\n✅ Nenhum dado RAW foi modificado."
)