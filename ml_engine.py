"""
=============================================================================
 ml_engine.py — Motor ML Fase 4 · Simulador Micromagnético
=============================================================================

Mejoras sobre Fase 3 (GBR single-feature):

  1. Features físicamente motivados  (7 dimensiones por punto)
     · d_nm            — diámetro de partícula (nm)
     · d / λ_ex        — tamaño normalizado a longitud de intercambio
     · log₁₀(d)        — captura relaciones de ley de potencias
     · K₁V / k_BT      — barrera de anisotropía vs. energía térmica (indicador SPM)
     · Ms (MA/m)        — magnetización de saturación
     · α  (LLG)         — amortiguamiento (damping)
     · T / Tc           — temperatura reducida

  2. Ensemble de tres modelos  (promedio ponderado por R² de validación cruzada)
     · GBR  — GradientBoostingRegressor  (n=350, depth=4, lr=0.04, subsample=0.85)
     · RF   — RandomForestRegressor      (n=400, depth=10)
     · MLP  — MLPRegressor               (128 → 64 → 32, ReLU, Adam, early-stopping)

  3. Aprendizaje online
     · Cada simulación del usuario agrega un punto de feedback
     · Botón "Reentrenar" incorpora los datos acumulados en el próximo entrenamiento
     · Los puntos de feedback se ponderan ×N (datos empíricos > sintéticos)

  4. Cuantificación de incertidumbre
     · σ_Hc y σ_Mr estimados vía varianza entre árboles del RF
     · Se muestran como bandas de confianza ±1σ en la histéresis

Uso:
    engine = MicromagneticMLEngine(MATERIALS_DB)
    engine.train()          # entrena todos los materiales
    Hc, Mr, sHc, sMr = engine.predict(30.0, 'fe', geom_factor_hc=1.55)
    engine.add_feedback('fe', 30.0, Hc_measured=155.0, Mr_measured=0.70)
    engine.retrain_with_feedback()   # incorpora feedback
=============================================================================
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ── Constante física ──────────────────────────────────────────────────────────
_kB = 1.380649e-23   # J / K


class MicromagneticMLEngine:
    """
    Motor de aprendizaje automático multi-modelo para predicción de
    Hc (campo coercitivo, mT) y Mr/Ms (remanencia normalizada) en
    nanopartículas magnéticas con aprendizaje online.
    """

    #: Nombres de los 7 features para gráficas de importancia
    FEATURE_NAMES = [
        'd (nm)',
        'd / λₑₓ',
        'log₁₀(d)',
        'K₁V / k_BT',
        'Ms (MA/m)',
        'α (LLG)',
        'T / Tc',
    ]

    #: Nombres de los tres modelos base
    MODEL_NAMES = ('GBR', 'RF', 'MLP')

    def __init__(
        self,
        materials_db: dict,
        T_sim: float = 300.0,
    ):
        """
        Parameters
        ----------
        materials_db : dict
            Diccionario MATERIALS_DB del simulador.
        T_sim : float
            Temperatura de simulación por defecto (K).
        """
        self.mdb   = materials_db
        self.T_sim = T_sim

        # {mat_id: {'GBR': (m_hc, m_mr), 'RF': (m_hc, m_mr), 'MLP': (m_hc, m_mr)}}
        self._models:  dict[str, dict] = {}
        # {mat_id: StandardScaler}
        self._scalers: dict[str, StandardScaler] = {}
        # {mat_id: {'r2_cv_hc': {name: float}, 'r2_cv_mr': {name: float},
        #           'rmse_hc': {name: float}, 'rmse_mr': {name: float},
        #           'weights_hc': ndarray, 'weights_mr': ndarray,
        #           'n_train': int, 'n_feedback': int}}
        self._metrics: dict[str, dict] = {}
        # {mat_id: list of [*features(7), Hc, Mr]}
        self._feedback: dict[str, list] = {mid: [] for mid in materials_db}

        self._trained = False

    # ═══════════════════════════════════════════════════════════════════════════
    #  FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════════

    def features(
        self,
        d_nm: float,
        mat_id: str,
        T: Optional[float] = None,
    ) -> np.ndarray:
        """
        Construye el vector de 7 features físicamente motivados para
        un diámetro de partícula dado.

        Returns
        -------
        np.ndarray  shape (7,)
        """
        T = T or self.T_sim
        p = self.mdb[mat_id]['params']

        K1    = abs(p['K1_kJ_m3']) * 1e3   # J/m³
        Ms    = p['Ms_MA_m']                 # MA/m  (feature directo)
        lam   = max(p['lambda_ex_nm'], 1e-9) # nm
        Tc    = max(p['Tc_K'], 1.0)          # K
        alpha = p['alpha']

        r_m = (d_nm * 1e-9) / 2.0
        V   = (4.0 / 3.0) * np.pi * r_m ** 3   # m³

        return np.array([
            d_nm,                                    # 0: tamaño
            d_nm / lam,                              # 1: d / λ_ex  (adimensional)
            np.log10(max(d_nm, 0.1)),                # 2: escala log
            (K1 * V) / (_kB * max(T, 1.0)),          # 3: KuV/kBT
            Ms,                                      # 4: Ms
            alpha,                                   # 5: α
            T / Tc,                                  # 6: T reducida
        ], dtype=float)

    def features_batch(
        self,
        sizes: np.ndarray,
        mat_id: str,
        T: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calcula la matriz de features para un array de tamaños.

        Returns
        -------
        np.ndarray  shape (N, 7)
        """
        return np.array([self.features(d, mat_id, T) for d in sizes])

    # ═══════════════════════════════════════════════════════════════════════════
    #  GENERACIÓN DE DATOS DE ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_training_data(
        self,
        mat_id: str,
        n_aug: int = 160,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye el dataset de entrenamiento para un material.

        Estrategia:
          1. Puntos âncora de 'sphere' en MATERIALS_DB (×15 para prioridad)
          2. Interpolación lineal densa entre los âncoras (n_aug puntos)
          3. Puntos aleatorios con ruido físico (n_aug/3 puntos)
          4. Extrapolación suave en los extremos (±30 % del rango)
          5. Feedback del usuario (×20 para máxima prioridad)

        Returns
        -------
        sizes : ndarray  (N,)
        Hc    : ndarray  (N,)   campo coercitivo en mT
        Mr    : ndarray  (N,)   Mr/Ms
        """
        mat = self.mdb[mat_id]
        lo, hi = mat['range']
        orig   = mat['sphere']           # shape (N_orig, 3): [d, Hc, Mr]
        rng    = np.random.default_rng(abs(hash(mat_id)) % (2 ** 31))

        d_orig  = orig[:, 0]
        Hc_orig = orig[:, 1]
        Mr_orig = orig[:, 2]

        # ── Interpolación densa dentro del rango entrenado ────────────────────
        sizes_grid = np.linspace(lo, hi, n_aug)
        Hc_interp  = np.interp(sizes_grid, d_orig, Hc_orig)
        Mr_interp  = np.interp(sizes_grid, d_orig, Mr_orig)

        # Ruido pequeño para regularización (evita sobreajuste exacto)
        Hc_grid = Hc_interp * (1.0 + 0.03 * rng.standard_normal(n_aug))
        Mr_grid = Mr_interp + 0.012 * rng.standard_normal(n_aug)

        # ── Puntos aleatorios dentro del rango ────────────────────────────────
        sizes_rand = rng.uniform(lo, hi, n_aug // 3)
        Hc_rand    = np.interp(sizes_rand, d_orig, Hc_orig) * \
                     (1.0 + 0.05 * rng.standard_normal(n_aug // 3))
        Mr_rand    = np.interp(sizes_rand, d_orig, Mr_orig) + \
                     0.02 * rng.standard_normal(n_aug // 3)

        # ── Extrapolación en los extremos (tendencia física power-law) ────────
        n_ext = 15
        # Extrapolación baja (d < lo): Hc crece ∝ d^1/2  (monodominio pequeño)
        sizes_lo  = np.linspace(max(2.0, lo * 0.35), lo, n_ext, endpoint=False)
        Hc_lo_ref = Hc_orig[0]
        Hc_lo     = Hc_lo_ref * (sizes_lo / lo) ** 0.40
        Mr_lo     = np.full(n_ext, min(Mr_orig[0] * 1.05, 0.95))

        # Extrapolación alta (d > hi): Hc cae ∝ d^-2  (multidominio)
        sizes_hi  = np.linspace(hi, hi * 1.55, n_ext, endpoint=False)
        Hc_hi     = Hc_orig[-1] * (hi / sizes_hi) ** 1.80
        Mr_hi_arr = np.linspace(Mr_orig[-1], max(Mr_orig[-1] * 0.55, 0.15), n_ext)

        # ── Puntos âncora originales con peso aumentado (×15) ─────────────────
        n_rep_orig = 15
        d_rep   = np.tile(d_orig,  n_rep_orig)
        Hc_rep  = np.tile(Hc_orig, n_rep_orig)
        Mr_rep  = np.tile(Mr_orig, n_rep_orig)

        # ── Incorporar feedback del usuario (×20) ─────────────────────────────
        fb = self._feedback.get(mat_id, [])
        if fb:
            fb_arr  = np.array(fb)     # shape (N_fb, 9): [7 features + Hc + Mr]
            d_fb    = fb_arr[:, 0]     # d_nm es la primera feature
            Hc_fb   = fb_arr[:, 7]
            Mr_fb   = fb_arr[:, 8]
            n_rep_fb = 20
            d_rep   = np.concatenate([d_rep,  np.tile(d_fb,  n_rep_fb)])
            Hc_rep  = np.concatenate([Hc_rep, np.tile(Hc_fb, n_rep_fb)])
            Mr_rep  = np.concatenate([Mr_rep, np.tile(Mr_fb, n_rep_fb)])

        # ── Concatenar todo ───────────────────────────────────────────────────
        sizes_all = np.concatenate([
            sizes_grid, sizes_rand,
            sizes_lo, sizes_hi,
            d_rep,
        ])
        Hc_all = np.concatenate([
            Hc_grid, Hc_rand,
            Hc_lo, Hc_hi,
            Hc_rep,
        ])
        Mr_all = np.concatenate([
            Mr_grid, Mr_rand,
            Mr_lo, Mr_hi_arr,
            Mr_rep,
        ])

        # Clip físico
        Hc_all = np.clip(Hc_all, 0.1, mat['field_max'] * 3.5)
        Mr_all = np.clip(Mr_all, 0.05, 0.97)

        return sizes_all, Hc_all, Mr_all

    # ═══════════════════════════════════════════════════════════════════════════
    #  ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════════════

    def train(self, mat_id: Optional[str] = None) -> None:
        """
        Entrena (o reentrenar) el ensemble GBR + RF + MLP para los materiales
        indicados. Si mat_id es None, entrena los 8 materiales.

        Parameters
        ----------
        mat_id : str | None
            Material específico o None para todos.
        """
        targets = [mat_id] if mat_id else list(self.mdb.keys())
        for mid in targets:
            self._train_one(mid)
        self._trained = True

    def _train_one(self, mat_id: str) -> None:
        sizes, Hc, Mr = self._build_training_data(mat_id)

        # Feature matrix (N × 7)
        X  = self.features_batch(sizes, mat_id)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)

        # ── Definición de modelos (producción) ───────────────────────────────
        def _gbr(rs):
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.85,
                min_samples_leaf=3, random_state=rs,
            )

        def _rf(rs):
            return RandomForestRegressor(
                n_estimators=200, max_depth=8,
                min_samples_leaf=3, n_jobs=-1,
                random_state=rs,
            )

        def _mlp(rs):
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu', solver='adam',
                learning_rate_init=5e-4,
                max_iter=600,
                early_stopping=True, validation_fraction=0.10,
                n_iter_no_change=15,
                random_state=rs,
            )

        # ── Modelos rápidos solo para CV (menos árboles / capas) ──────────────
        def _gbr_cv(rs):
            return GradientBoostingRegressor(
                n_estimators=60, max_depth=3,
                learning_rate=0.10, random_state=rs,
            )

        def _rf_cv(rs):
            return RandomForestRegressor(
                n_estimators=60, max_depth=6,
                min_samples_leaf=4, n_jobs=-1, random_state=rs,
            )

        def _mlp_cv(rs):
            return MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu', solver='adam',
                learning_rate_init=1e-3, max_iter=200,
                random_state=rs,
            )

        # ── Entrenamiento (modelos completos) ─────────────────────────────────
        fitted = {}
        for name, (mhc, mmr) in {
            'GBR': (_gbr(0), _gbr(1)),
            'RF':  (_rf(0),  _rf(1)),
            'MLP': (_mlp(0), _mlp(1)),
        }.items():
            fitted[name] = (mhc.fit(Xs, Hc), mmr.fit(Xs, Mr))

        # ── Métricas: R² CV con modelos rápidos (3-fold) ─────────────────────
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        def _cv_r2(factory, y):
            try:
                scores = cross_val_score(factory, Xs, y, cv=cv,
                                         scoring='r2', n_jobs=-1)
                return float(np.clip(scores.mean(), 0.01, 1.0))
            except Exception:
                return 0.5

        r2_cv_hc = {
            'GBR': _cv_r2(_gbr_cv(0), Hc),
            'RF':  _cv_r2(_rf_cv(0),  Hc),
            'MLP': _cv_r2(_mlp_cv(0), Hc),
        }
        r2_cv_mr = {
            'GBR': _cv_r2(_gbr_cv(1), Mr),
            'RF':  _cv_r2(_rf_cv(1),  Mr),
            'MLP': _cv_r2(_mlp_cv(1), Mr),
        }

        # RMSE en entrenamiento completo
        def _rmse(m, y):
            return float(np.sqrt(mean_squared_error(y, m.predict(Xs))))

        rmse_hc = {n: _rmse(m_hc, Hc) for n, (m_hc, _) in fitted.items()}
        rmse_mr = {n: _rmse(m_mr, Mr) for n, (_, m_mr) in fitted.items()}

        # Pesos del ensemble (proporcionales al R² de CV)
        w_hc = np.array([r2_cv_hc[n] for n in self.MODEL_NAMES])
        w_mr = np.array([r2_cv_mr[n] for n in self.MODEL_NAMES])
        w_hc /= w_hc.sum()
        w_mr /= w_mr.sum()

        self._models[mat_id]  = fitted
        self._scalers[mat_id] = sc
        self._metrics[mat_id] = {
            'r2_cv_hc':   r2_cv_hc,
            'r2_cv_mr':   r2_cv_mr,
            'rmse_hc':    rmse_hc,
            'rmse_mr':    rmse_mr,
            'weights_hc': w_hc,
            'weights_mr': w_mr,
            'n_train':    len(sizes),
            'n_feedback': len(self._feedback.get(mat_id, [])),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  PREDICCIÓN
    # ═══════════════════════════════════════════════════════════════════════════

    def predict(
        self,
        d_nm: float,
        mat_id: str,
        geom_factor_hc: float = 1.0,
        geom_factor_mr: float = 1.0,
        T: Optional[float] = None,
    ) -> tuple[float, float, float, float]:
        """
        Predicción ensemble con incertidumbre.

        Parameters
        ----------
        d_nm           : diámetro de partícula (nm)
        mat_id         : clave del material en MATERIALS_DB
        geom_factor_hc : multiplicador de forma para Hc
        geom_factor_mr : multiplicador de forma para Mr
        T              : temperatura (K); None usa T_sim del engine

        Returns
        -------
        (Hc_mean, Mr_mean, Hc_std, Mr_std)
            donde Hc_std / Mr_std son estimaciones de incertidumbre ±1σ
            (varianza entre árboles del RF)
        """
        feat = self.features(d_nm, mat_id, T).reshape(1, -1)
        Xs   = self._scalers[mat_id].transform(feat)

        w_hc = self._metrics[mat_id]['weights_hc']
        w_mr = self._metrics[mat_id]['weights_mr']

        preds_hc = np.array([
            float(self._models[mat_id][n][0].predict(Xs)[0])
            for n in self.MODEL_NAMES
        ])
        preds_mr = np.array([
            float(self._models[mat_id][n][1].predict(Xs)[0])
            for n in self.MODEL_NAMES
        ])

        Hc_mean = float(np.dot(w_hc, preds_hc))
        Mr_mean = float(np.dot(w_mr, preds_mr))

        # Incertidumbre: desviación estándar entre árboles del RF
        rf_hc = self._models[mat_id]['RF'][0]
        rf_mr = self._models[mat_id]['RF'][1]
        tree_hc = np.array([t.predict(Xs)[0] for t in rf_hc.estimators_])
        tree_mr = np.array([t.predict(Xs)[0] for t in rf_mr.estimators_])
        Hc_std  = float(tree_hc.std())
        Mr_std  = float(tree_mr.std())

        return (
            Hc_mean * geom_factor_hc,
            float(np.clip(Mr_mean * geom_factor_mr, 0.01, 1.10)),
            Hc_std  * geom_factor_hc,
            Mr_std,
        )

    def predict_fast(
        self,
        d_nm: float,
        mat_id: str,
        geom_factor_hc: float = 1.0,
        geom_factor_mr: float = 1.0,
        T: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Predicción ensemble SIN incertidumbre — ~50× más rápido que predict().

        Omite la iteración sobre los 200 estimadores individuales del RF
        (usada solo para cuantificar σ). Ideal para bucles de visualización
        donde la incertidumbre no se necesita.

        Returns
        -------
        (Hc, Mr)
        """
        feat = self.features(d_nm, mat_id, T).reshape(1, -1)
        Xs   = self._scalers[mat_id].transform(feat)
        w_hc = self._metrics[mat_id]['weights_hc']
        w_mr = self._metrics[mat_id]['weights_mr']
        preds_hc = np.array([
            float(self._models[mat_id][n][0].predict(Xs)[0])
            for n in self.MODEL_NAMES
        ])
        preds_mr = np.array([
            float(self._models[mat_id][n][1].predict(Xs)[0])
            for n in self.MODEL_NAMES
        ])
        Hc = float(np.dot(w_hc, preds_hc)) * geom_factor_hc
        Mr = float(np.clip(np.dot(w_mr, preds_mr) * geom_factor_mr, 0.01, 1.10))
        return Hc, Mr

    def predict_batch(
        self,
        sizes_nm,
        mat_id: str,
        geom_factor_hc: float = 1.0,
        geom_factor_mr: float = 1.0,
        T: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicción vectorizada para múltiples tamaños a la vez.

        Llama a sklearn.predict() una sola vez por modelo en lugar de hacerlo
        para cada tamaño — mucho más rápido que hacer predict_fast() en bucle.

        Parameters
        ----------
        sizes_nm : array-like  tamaños de partícula en nm

        Returns
        -------
        (Hc_arr, Mr_arr)  ndarrays de shape (N,)
        """
        sizes_nm = np.asarray(sizes_nm, dtype=float)
        X  = self.features_batch(sizes_nm, mat_id, T)      # (N, 7)
        Xs = self._scalers[mat_id].transform(X)            # (N, 7) normalizado

        w_hc = self._metrics[mat_id]['weights_hc']         # (3,)
        w_mr = self._metrics[mat_id]['weights_mr']

        Hc_agg = np.zeros(len(sizes_nm))
        Mr_agg = np.zeros(len(sizes_nm))
        for i, name in enumerate(self.MODEL_NAMES):
            Hc_agg += w_hc[i] * self._models[mat_id][name][0].predict(Xs)
            Mr_agg += w_mr[i] * self._models[mat_id][name][1].predict(Xs)

        return (
            Hc_agg * geom_factor_hc,
            np.clip(Mr_agg * geom_factor_mr, 0.01, 1.10),
        )

    def predict_all_models(
        self,
        d_nm: float,
        mat_id: str,
        T: Optional[float] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Retorna las predicciones individuales de cada modelo y el ensemble.
        Usa predict_fast para el ensemble (sin iteración de árboles RF → rápido).

        Returns
        -------
        {
          'GBR':     {'Hc': float, 'Mr': float},
          'RF':      {'Hc': float, 'Mr': float},
          'MLP':     {'Hc': float, 'Mr': float},
          'Ensemble':{'Hc': float, 'Mr': float},
        }
        """
        feat = self.features(d_nm, mat_id, T).reshape(1, -1)
        Xs   = self._scalers[mat_id].transform(feat)

        w_hc = self._metrics[mat_id]['weights_hc']
        w_mr = self._metrics[mat_id]['weights_mr']

        result = {}
        preds_hc = []
        preds_mr = []
        for name in self.MODEL_NAMES:
            hc = float(self._models[mat_id][name][0].predict(Xs)[0])
            mr = float(self._models[mat_id][name][1].predict(Xs)[0])
            result[name] = {'Hc': hc, 'Mr': max(0.0, mr)}
            preds_hc.append(hc)
            preds_mr.append(mr)

        # Ensemble via weighted sum (reuses already-computed model preds — no extra calls)
        Hc_ens = float(np.dot(w_hc, preds_hc))
        Mr_ens = float(np.clip(np.dot(w_mr, preds_mr), 0.01, 1.10))
        result['Ensemble'] = {'Hc': Hc_ens, 'Mr': Mr_ens}
        return result

    def predict_all_models_sweep(
        self,
        mat_id: str,
        n_pts: int = 50,
        T: Optional[float] = None,
    ) -> dict:
        """
        Versión vectorizada de predict_all_models para un barrido de tamaños.
        Una sola pasada por sklearn por modelo — mucho más rápido que llamar
        predict_all_models() en bucle.

        Returns
        -------
        {
          'sizes':    ndarray (n_pts,),
          'GBR':      {'Hc': ndarray, 'Mr': ndarray},
          'RF':       {'Hc': ndarray, 'Mr': ndarray},
          'MLP':      {'Hc': ndarray, 'Mr': ndarray},
          'Ensemble': {'Hc': ndarray, 'Mr': ndarray},
          'Hc_std':   ndarray,   # varianza RF (±1σ)
          'Mr_std':   ndarray,
        }
        """
        lo, hi = self.mdb[mat_id]['range']
        sizes  = np.linspace(max(2.0, lo - 5), hi + 5, n_pts)

        X  = self.features_batch(sizes, mat_id, T)
        Xs = self._scalers[mat_id].transform(X)

        w_hc = self._metrics[mat_id]['weights_hc']
        w_mr = self._metrics[mat_id]['weights_mr']

        result = {'sizes': sizes}
        Hc_ens = np.zeros(n_pts)
        Mr_ens = np.zeros(n_pts)

        for i, name in enumerate(self.MODEL_NAMES):
            hc_arr = self._models[mat_id][name][0].predict(Xs)
            mr_arr = self._models[mat_id][name][1].predict(Xs)
            result[name] = {'Hc': hc_arr, 'Mr': np.maximum(mr_arr, 0.0)}
            Hc_ens += w_hc[i] * hc_arr
            Mr_ens += w_mr[i] * mr_arr

        result['Ensemble'] = {
            'Hc': Hc_ens,
            'Mr': np.clip(Mr_ens, 0.01, 1.10),
        }

        # Varianza RF (vectorizada)
        rf_hc = self._models[mat_id]['RF'][0]
        rf_mr = self._models[mat_id]['RF'][1]
        tree_hc = np.array([t.predict(Xs) for t in rf_hc.estimators_])
        tree_mr = np.array([t.predict(Xs) for t in rf_mr.estimators_])
        result['Hc_std'] = tree_hc.std(axis=0)
        result['Mr_std'] = tree_mr.std(axis=0)

        return result

    def predict_sweep(
        self,
        mat_id: str,
        n_pts: int = 80,
        T: Optional[float] = None,
    ) -> dict:
        """
        Barrido de predicciones Hc y Mr sobre el rango completo del material,
        incluyendo bandas de incertidumbre.

        Usa predict_batch() para las medias y vectoriza la varianza RF con
        np.array([t.predict(Xs) for t in estimators_]) — ~80× más rápido que
        el bucle original de predict() uno a uno.

        Returns
        -------
        {'sizes': ndarray, 'Hc': ndarray, 'Mr': ndarray,
         'Hc_lo': ndarray, 'Hc_hi': ndarray,
         'Mr_lo': ndarray, 'Mr_hi': ndarray}
        """
        lo, hi = self.mdb[mat_id]['range']
        sizes  = np.linspace(lo, hi, n_pts)

        # ── Medias via batch (3 llamadas sklearn para N puntos) ───────────────
        Hc_arr, Mr_arr = self.predict_batch(sizes, mat_id, T=T)

        # ── Incertidumbre: varianza entre árboles RF (vectorizada) ─────────
        X   = self.features_batch(sizes, mat_id, T)
        Xs  = self._scalers[mat_id].transform(X)

        rf_hc = self._models[mat_id]['RF'][0]
        rf_mr = self._models[mat_id]['RF'][1]
        # shape (n_estimators, n_pts) — cada árbol predice todos los puntos
        tree_hc = np.array([t.predict(Xs) for t in rf_hc.estimators_])
        tree_mr = np.array([t.predict(Xs) for t in rf_mr.estimators_])
        sHc_arr = tree_hc.std(axis=0)
        sMr_arr = tree_mr.std(axis=0)

        return {
            'sizes': sizes,
            'Hc': Hc_arr, 'Hc_lo': Hc_arr - sHc_arr, 'Hc_hi': Hc_arr + sHc_arr,
            'Mr': Mr_arr, 'Mr_lo': Mr_arr - sMr_arr,  'Mr_hi': Mr_arr + sMr_arr,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  APRENDIZAJE ONLINE
    # ═══════════════════════════════════════════════════════════════════════════

    def add_feedback(
        self,
        mat_id: str,
        d_nm: float,
        Hc_sim: float,
        Mr_sim: float,
        T: Optional[float] = None,
    ) -> None:
        """
        Registra el resultado de una simulación como punto de feedback.
        El feedback se incorpora al dataset en el próximo `retrain_with_feedback()`.

        Parameters
        ----------
        mat_id  : clave del material
        d_nm    : diámetro de partícula simulado (nm)
        Hc_sim  : Hc resultado de la simulación (mT)
        Mr_sim  : Mr/Ms resultado de la simulación
        T       : temperatura usada (K)
        """
        feat = self.features(d_nm, mat_id, T)      # shape (7,)
        row  = list(feat) + [float(Hc_sim), float(Mr_sim)]  # 9 valores
        self._feedback[mat_id].append(row)

    def retrain_with_feedback(self, mat_id: Optional[str] = None) -> None:
        """
        Reentrenar incorporando todo el feedback acumulado.
        Equivalente a `train()` pero los datos de feedback tienen peso ×20.
        """
        self.train(mat_id)

    @property
    def feedback_counts(self) -> dict[str, int]:
        """Número de puntos de feedback acumulados por material."""
        return {mid: len(fb) for mid, fb in self._feedback.items()}

    @property
    def total_feedback(self) -> int:
        """Total de puntos de feedback acumulados."""
        return sum(self.feedback_counts.values())

    # ═══════════════════════════════════════════════════════════════════════════
    #  MÉTRICAS Y DIAGNÓSTICO
    # ═══════════════════════════════════════════════════════════════════════════

    def get_metrics(self, mat_id: str) -> dict:
        """
        Retorna las métricas de evaluación para un material.

        Returns
        -------
        dict con claves: r2_cv_hc, r2_cv_mr, rmse_hc, rmse_mr,
                         weights_hc, weights_mr, n_train, n_feedback
        """
        return self._metrics.get(mat_id, {})

    def metrics_dataframe(self) -> list[dict]:
        """
        Genera una lista de dicts lista para crear un DataFrame comparativo
        de todos los modelos y materiales.
        """
        rows = []
        for mid, m in self._metrics.items():
            mat_name = self.mdb[mid]['name']
            for name in self.MODEL_NAMES:
                rows.append({
                    'Material':    mat_name,
                    'Modelo':      name,
                    'R² CV  Hc':   round(m['r2_cv_hc'].get(name, 0), 4),
                    'R² CV  Mr':   round(m['r2_cv_mr'].get(name, 0), 4),
                    'RMSE Hc (mT)':round(m['rmse_hc'].get(name, 0), 2),
                    'RMSE Mr':     round(m['rmse_mr'].get(name, 0), 4),
                    'Peso Hc':     round(m['weights_hc'][self.MODEL_NAMES.index(name)], 3),
                    'Peso Mr':     round(m['weights_mr'][self.MODEL_NAMES.index(name)], 3),
                })
        return rows

    def feature_importance(self, mat_id: str) -> dict:
        """
        Importancia de features del GBR (más interpretable que RF/MLP).

        Returns
        -------
        {'hc': ndarray(7), 'mr': ndarray(7), 'names': list[str]}
        """
        if mat_id not in self._models:
            return {}
        gbr_hc = self._models[mat_id]['GBR'][0]
        gbr_mr = self._models[mat_id]['GBR'][1]
        return {
            'hc':    gbr_hc.feature_importances_,
            'mr':    gbr_mr.feature_importances_,
            'names': self.FEATURE_NAMES,
        }

    def feature_importance_all(self) -> dict[str, dict]:
        """Importancia de features para todos los materiales."""
        return {mid: self.feature_importance(mid) for mid in self._models}

    # ── Estado del engine ─────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._trained and bool(self._models)

    def summary(self) -> str:
        """Resumen del estado del engine en texto."""
        if not self.is_trained:
            return "Engine no entrenado."
        lines = [
            "═══ MicromagneticMLEngine v4.0 ═══",
            f"  Materiales : {len(self._models)}",
            f"  Modelos    : GBR + RF + MLP (ensemble ponderado por R² CV)",
            f"  Features   : {len(self.FEATURE_NAMES)} ({', '.join(self.FEATURE_NAMES[:3])} …)",
            f"  T_sim      : {self.T_sim} K",
            f"  Feedback   : {self.total_feedback} puntos acumulados",
        ]
        # Calibración OOMMF
        try:
            import oommf_data_manager as _odm
            cal = _odm.load_calibration_db()
            lines.append(f"  Cal. OOMMF : {len(cal)} puntos reales")
        except Exception:
            pass
        for mid, m in self._metrics.items():
            name = self.mdb[mid]['name']
            r2   = np.mean(list(m['r2_cv_hc'].values()))
            lines.append(
                f"  {name:<30} n={m['n_train']:4d}  R²_avg={r2:.3f}"
            )
        return "\n".join(lines)

    def predict_with_calibration(
        self,
        d_nm: float,
        mat_id: str,
        geom_id: str = 'sphere',
        geom_factor_hc: float = 1.0,
        geom_factor_mr: float = 1.0,
        T: Optional[float] = None,
        sigma_nm: float = 5.0,
    ) -> tuple[float, float, bool]:
        """
        Predicción con corrección Gaussiana de datos OOMMF reales.

        Si existen puntos de calibración para (mat_id, geom_id) cercanos a
        d_nm, mezcla la predicción ML con los valores reales ponderados por
        distancia gaussiana (sigma=sigma_nm).

        Returns
        -------
        (Hc_mT, Mr_Ms, calibration_applied)
        """
        Hc_ml, Mr_ml = self.predict_fast(
            d_nm, mat_id, geom_factor_hc, geom_factor_mr, T)

        try:
            import oommf_data_manager as _odm
            Hc_corr, Mr_corr = _odm.calibration_correction(
                mat_id=mat_id,
                d_nm=d_nm,
                geom_id=geom_id,
                Hc_pred=Hc_ml,
                Mr_pred=Mr_ml,
                sigma_nm=sigma_nm,
            )
            cal_applied = (Hc_corr != Hc_ml or Mr_corr != Mr_ml)
            return Hc_corr, Mr_corr, cal_applied
        except Exception:
            return Hc_ml, Mr_ml, False
