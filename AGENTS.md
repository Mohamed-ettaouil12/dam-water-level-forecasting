# AGENTS.md — Règles obligatoires pour les agents

## Environnement Python (CRITIQUE)

**Toujours utiliser `.venv/bin/python` et `.venv/bin/pip`.**

- Le venv `.venv` utilise **Python 3.11.9** (requis pour torch CUDA)
- Le `python3` système est **Python 3.14** — **NE JAMAIS l'utiliser**
- Ne JAMAIS recréer le venv avec `python3` (c'est 3.14)
- Ne JAMAIS lancer `python3` ou `pip` nus (sans le chemin `.venv/bin/`)

```bash
# ✅ CORRECT
.venv/bin/python -m src.models.baseline_xgboost
.venv/bin/pip install package

# ❌ INTERDIT
python3 -m src.models.baseline_xgboost
pip install package
python3 -m pytest
```

## Commandes autorisées

| Action | Commande |
|--------|----------|
| Lancer un script | `.venv/bin/python -m module` |
| Tests | `.venv/bin/python -m pytest` |
| Installer package | `.venv/bin/pip install package` |
| Python interactif | `.venv/bin/python` |

## Vérification rapide

```bash
.venv/bin/python --version  # Doit afficher 3.11.x
```