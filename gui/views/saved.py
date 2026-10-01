"""🖼️ Figuras and 💾 Modelos guardados: what earlier runs left on disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from session import ROOT, available_models, ss
from ui import empty

from tirepinn.config import Config
from tirepinn.pinn import TirePINN


def render_figures() -> None:
    dirs = []
    if ss.result is not None:
        dirs.append(Path(ss.result.out_dir).resolve())
    if ss.loaded is not None:
        dirs.append(Path(ss.loaded["dir"]).resolve())
    dirs += [p for p in available_models() if p.resolve() not in dirs]
    if not dirs:
        empty("🖼️", "Todavía no hay figuras", "Aparecen al terminar un entrenamiento.")
        return
    labels = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in dirs]
    folder = dirs[labels.index(st.selectbox("Carpeta", labels))]
    captions = {
        "01_stints.png": "Predicciones de cada modelo sobre stints de prueba",
        "02_extrapolation.png": "Extrapolación más allá de lo observado",
        "03_latent_states.png": "Estados latentes θ y d",
        "04_parameters.png": "Convergencia de los parámetros físicos",
        "05_loss.png": "Términos de la función de pérdida",
        "06_cliff_map.png": "Mapa de decisión por compuesto",
    }
    images = sorted(folder.glob("*.png"))
    if not images:
        st.info("Esta carpeta no tiene figuras.")
    for img in images:
        st.image(str(img), caption=captions.get(img.name, img.name), width="stretch")
        st.download_button("⬇️ " + img.name, img.read_bytes(), img.name, "image/png",
                           key=f"dl-{folder}-{img.name}")
    for extra in ("report.txt", "pinn_params.json"):
        path = folder / extra
        if path.exists():
            with st.expander(extra):
                text = path.read_text(encoding="utf-8")
                st.code(json.dumps(json.loads(text), indent=2) if extra.endswith(".json") else text,
                        language=None)


def render_models() -> None:
    st.write("Abre un modelo entrenado antes (por la interfaz o por `run_train.py`) para usarlo "
             "en **Estrategia** sin volver a entrenar.")
    options = available_models()
    if not options:
        empty("💾", "No hay modelos guardados",
              "Cada entrenamiento se guarda en <code>outputs/</code> y aparecerá aquí.")
        return
    rel = [str(p.relative_to(ROOT)) for p in options]
    c1, c2 = st.columns([3, 1])
    chosen = c1.selectbox("Modelo", rel)
    if c2.button("📂 Cargar modelo", width="stretch"):
        folder = ROOT / chosen
        cfg_path = folder / "config.json"
        cfg = Config.from_json(str(cfg_path)) if cfg_path.exists() else Config()
        ss.loaded = {"pinn": TirePINN.load(folder, cfg), "cfg": cfg, "dir": folder}
        st.toast(f"Modelo cargado: {chosen}")

    loaded = ss.loaded
    if loaded is None:
        return
    folder, cfg, pinn = loaded["dir"], loaded["cfg"], loaded["pinn"]
    st.success(f"Modelo activo: `{folder.relative_to(ROOT)}` · fuente {cfg.data.source}")
    c1, c2 = st.columns(2)
    params = pinn.learned_params().as_dict()
    c1.dataframe(pd.DataFrame(
        [{"Parámetro": n, "Valor": v, "Estado": "estimado" if n in cfg.pinn.free_params else "fijo"}
         for n, v in params.items() if n != "p"]
    ).style.format(precision=4), width="stretch", hide_index=True)
    report = folder / "report.txt"
    if report.exists():
        c2.code(report.read_text(encoding="utf-8"), language=None)
    with st.expander("config.json"):
        st.json(asdict(cfg), expanded=False)
