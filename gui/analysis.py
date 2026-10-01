"""Checks and per-stint numbers behind the results views. No Streamlit here."""

from __future__ import annotations

import numpy as np

from tirepinn.config import Config


def race_names(cfg: Config) -> dict[str, str]:
    """Stint-id prefix -> race name, as `build_multi_dataset` builds the prefix."""
    return {gp[:3].upper(): gp for gp in (cfg.data.gps or (cfg.data.gp,))}


def race_of(stint, names: dict[str, str], fallback: str) -> str:
    """Multi-race ids look like SAU-PIA-S1; single-race ids (PIA-S1) carry no race."""
    parts = stint.stint_id.split("-")
    if len(parts) >= 3:
        return names.get(parts[0], parts[0])
    return fallback


def stint_errors(stint, models: dict) -> dict[str, float]:
    """RMSE of each model over this stint's observed laps."""
    out = {}
    for name, model in models.items():
        pred = np.asarray(model.predict_stint(stint.context, stint.laps)).ravel()
        out[name] = float(np.sqrt(np.mean((pred - stint.delta) ** 2)))
    return out


def diagnose(res) -> list[tuple[str, str, str]]:
    """Plain checks that catch a training run that did not converge.

    Returns (level, title, explanation) with level "ok", "warn" or "bad".
    """
    out = []
    cfg, phys = res.cfg, res.cfg.physics
    pinn = res.models["PINN"]
    params = pinn.learned_params().as_dict()
    real = cfg.data.source == "fastf1"
    metrics = {m.name: m for m in res.metrics}
    p = metrics.get("PINN")

    if cfg.pinn.adam_iters < 5000:
        out.append(("warn", f"Entrenamiento corto: {cfg.pinn.adam_iters:,} iteraciones de Adam",
                    "La física se impone poco a poco. Para resultados fiables usa el preajuste "
                    "**Estándar** (15.000 + 3.000)."))

    bounds = {"gamma1": phys.gamma1_bounds, "gamma2": phys.gamma2_bounds,
              "kappa": phys.kappa_bounds}
    pinned = [n for n, (lo, hi) in bounds.items() if n in cfg.pinn.free_params
              and min(abs(params[n] - lo), abs(params[n] - hi)) < 0.02 * (hi - lo)]
    if pinned:
        out.append(("bad", f"Parámetros pegados a su límite: {', '.join(pinned)}",
                    "El optimizador empujó estos coeficientes hasta su cota: la solución no es "
                    "física. Con datos reales deja libres solo **kw** y **kappa**."))
    if real and {"gamma1", "gamma2"} & set(cfg.pinn.free_params):
        out.append(("warn", "gamma libre con datos reales",
                    "La escala del desgaste no es identificable desde tiempos de vuelta "
                    "(ver README): con gamma libre la pérdida máxima puede crecer sin control. "
                    "Lo recomendado es estimar solo **kw** y **kappa**."))

    delta_max = params["gamma1"] + params["gamma2"]
    saturated = 0
    for s in res.test.stints:
        pred = np.asarray(pinn.predict_stint(s.context, s.laps)).ravel()
        if pred.max() > 0.9 * delta_max and s.delta.max() < 0.5 * delta_max:
            saturated += 1
    if saturated:
        out.append(("bad", f"La PINN predice un cliff que no existe en {saturated} de "
                           f"{len(res.test.stints)} stints de prueba",
                    f"La curva se dispara hasta su máximo ({delta_max:.1f} s) cuando los datos no "
                    "pasan de la mitad. Señal típica de falta de entrenamiento o de parámetros "
                    "mal identificados."))

    lin = metrics.get("Linear (classic)")
    if p and lin and p.rmse > lin.rmse:
        out.append(("bad", f"La PINN ({p.rmse:.3f} s) es peor que la regresión lineal ({lin.rmse:.3f} s)",
                    "Un modelo que no supera a una recta no ha aprendido la física."))
    if p and p.violation_rate > 0.01:
        out.append(("warn", f"{100 * p.violation_rate:.1f} % de vueltas donde el neumático «recupera» agarre",
                    "Físicamente imposible: la red aún no respeta bien la ecuación de desgaste."))

    if real:
        obs = np.concatenate([s.delta for s in res.data.stints])
        neg = float(np.mean(obs < -0.3))
        if neg > 0.15:
            out.append(("warn", f"{100 * neg:.0f} % de las vueltas observadas tienen pérdida negativa",
                        "Tras corregir combustible y evolución de pista, muchas vueltas salen *más "
                        "rápidas* que al inicio del stint. Ningún modelo de desgaste puede predecir "
                        "eso: suele indicar una carrera con mucha gestión de neumáticos o tráfico."))
    if not out:
        out.append(("ok", "Sin señales de mala convergencia",
                    "Parámetros dentro de sus límites, sin cliffs inventados y mejor que la "
                    "regresión lineal."))
    return out
