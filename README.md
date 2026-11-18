# DermaLens · Classification de lésions cutanées (HAM10000)

Projet de module **Deep Learning Avancé** (Cycle Ingénieur IA & Data Science), **Axe 1 : CNN avancé & Transfer Learning + Grad-CAM**.

Le projet classe 7 types de lésions pigmentées à partir d'images dermatoscopiques du dataset HAM10000. Le modèle est un CNN pré-entraîné (EfficientNetV2 / ConvNeXt) fine-tuné, exporté en ONNX et servi par une API FastAPI. L'interface React affiche des cartes **Grad-CAM** pour l'explicabilité. Le cycle de vie complet est outillé : DVC, MLflow, Docker et GitHub Actions.

> ⚠️ Outil pédagogique. Il ne fournit aucun diagnostic médical.

```
Navigateur ──► Frontend React (nginx :3000) ──/api──► FastAPI (:8000) ──► ONNX Runtime (logits + Grad-CAM)
                                                        ▲
              DVC (data/, models/) ──► MLflow Tracking & Model Registry (:5000)
```

---

## 0. Démarrage pour l'équipe

| Ressource | URL |
|---|---|
| Code (GitHub) | https://github.com/kaouech-rahim/ham10000-skin-lesion |
| Données et modèles (DagsHub, remote DVC) | https://dagshub.com/kaouech-rahim/ham10000-skin-lesion |
| Historique des runs (MLflow sur DagsHub) | https://dagshub.com/kaouech-rahim/ham10000-skin-lesion.mlflow |

### 0.1 Prérequis

- Git
- Python 3.11
- Docker Desktop, **démarré**
- DVC avec le support S3 : `pip install "dvc[s3]"`
- Être **collaborateur du dépôt DagsHub** : demandez une invitation à `kaouech-rahim`.

### 0.2 Récupérer le code

```bash
git clone https://github.com/kaouech-rahim/ham10000-skin-lesion.git
cd ham10000-skin-lesion
```

### 0.3 Configurer le remote DVC avec VOS identifiants DagsHub

Sur DagsHub, ouvrez *Remote → Data → DVC*, puis copiez votre token. Chaque membre utilise **son propre** token.

```bash
dvc remote modify origin --local access_key_id <votre_token_dagshub>
dvc remote modify origin --local secret_access_key <votre_token_dagshub>
```

L'option `--local` écrit dans `.dvc/config.local`, qui est ignoré par Git. **Ne commitez jamais votre token.**

### 0.4 Récupérer le modèle

```bash
dvc pull models/model.onnx.dvc models/convnext-focal.pt.dvc
```

Cette commande télécharge environ 210 Mo : `model.onnx` (le modèle servi par l'API) et `convnext-focal.pt` (le checkpoint PyTorch). `models/metadata.json` est déjà dans Git.

> Un `dvc pull` **sans argument** télécharge en plus tout le dataset (environ 7 Go : `data/raw` et `data/processed`). Ce n'est nécessaire que pour **réentraîner** le modèle.

### 0.5 Lancer l'application

```bash
docker compose up --build
```

Le premier build prend plusieurs minutes, le temps de télécharger les images Python, Node, nginx et MLflow.

| Service | URL |
|---|---|
| Application web | http://localhost:3000 |
| API + Swagger | http://localhost:8000/docs |
| MLflow local | http://localhost:5000 |

> Le MLflow local du port 5000 démarre **vide** : il ne contient pas les runs d'entraînement. L'historique complet des runs est sur DagsHub (lien MLflow ci-dessus).

### 0.6 Vérifier que tout fonctionne

1. http://localhost:8000/health renvoie `"status": "ok"` et `"model_loaded": true`.
2. Sur http://localhost:3000, le panneau du modèle affiche l'architecture `convnext_tiny`.
3. Choisissez une image dans la **galerie d'exemples** (ou déposez un JPG). Vous devez obtenir les probabilités des 7 classes **et** la carte Grad-CAM.

### 0.7 Problèmes fréquents

| Symptôme | Cause | Solution |
|---|---|---|
| `Cannot connect to the Docker daemon` / `error during connect` | Docker Desktop n'est pas lancé | Démarrer Docker Desktop, attendre « Engine running », puis relancer. |
| `Bind for 0.0.0.0:3000 failed: port is already allocated` (ou 8000, 5000) | Un autre programme utilise le port | Trouver le processus (`netstat -ano \| findstr :3000` sous Windows, `lsof -i :3000` sous macOS/Linux) et l'arrêter, ou changer le port à gauche dans `docker-compose.yml` (ex. `"3001:80"`). |
| `dvc pull` : `403 Forbidden`, `Unable to locate credentials` ou `access denied` | Identifiants DagsHub absents ou invalides, ou vous n'êtes pas collaborateur | Refaire l'étape 0.3 avec votre token, et vérifier votre invitation sur DagsHub. |
| `/health` renvoie `"degraded"`, les prédictions renvoient `503` | `models/model.onnx` est absent : `dvc pull` n'a pas été fait | Faire l'étape 0.4, puis `docker compose restart backend`. |
| Le build est très lent ou semble bloqué | Le premier téléchargement des images Docker est long | Patienter. `docker compose --progress plain up --build` affiche l'avancement. |

---

## 1. Démarrage rapide (1 commande)

Prérequis : Docker, et le modèle dans `models/`. Voir le §0 pour la configuration complète.

```bash
dvc pull models/model.onnx.dvc   # récupère le modèle servi par l'API
docker compose up --build        # (ou docker-compose up --build)
```

| Service | URL |
|---|---|
| Application web | http://localhost:3000 |
| API + Swagger | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

Sans modèle, l'API démarre en mode **dégradé** : `/health` renvoie `"degraded"` et les routes d'inférence renvoient `503`.

Pour servir le modèle quantifié INT8 : `MODEL_VARIANT=int8 docker compose up`.

---

## 2. Le dataset HAM10000 et les choix anti-fuite

10 015 images dermatoscopiques réparties en 7 classes très déséquilibrées :

| Code | Classe | Maligne |
|---|---|---|
| `nv` | Nævus mélanocytaire (~67 %) | |
| `mel` | Mélanome | ✔ |
| `bkl` | Lésion kératosique bénigne | |
| `bcc` | Carcinome basocellulaire | ✔ |
| `akiec` | Kératose actinique / Bowen | ✔ (pré-maligne) |
| `vasc` | Lésion vasculaire | |
| `df` | Dermatofibrome | |

**Les points méthodologiques à défendre en soutenance :**

1. **Split groupé par `lesion_id`.** Une même lésion peut avoir plusieurs photos. Le script `src/data/prepare.py` utilise `StratifiedGroupKFold` (≈ 70/15/15) et vérifie par une assertion qu'aucune lésion n'est partagée entre train, val et test. C'est testé dans `tests/test_prepare.py`.
2. **Pas de fuite par le prétraitement.** La normalisation utilise les statistiques ImageNet (fixes). L'augmentation n'est appliquée qu'au train.
3. **Métriques adaptées au déséquilibre.** Prédire toujours `nv` donne déjà ~67 % d'accuracy, ce qui la rend trompeuse. On suit donc :
   - la **balanced accuracy** (métrique officielle ISIC 2018), qui sert aussi à sélectionner le modèle ;
   - le **F1 macro** ;
   - le **rappel par classe**, en particulier `recall_mel` ;
   - l'**AUC one-vs-rest** et la matrice de confusion.
4. Le jeu de test officiel ISIC n'est pas public. On évalue donc sur notre propre test groupé, qui n'est touché qu'une seule fois, par `evaluate.py`.

### Téléchargement

```bash
pip install kaggle   # puis placer ~/.kaggle/kaggle.json
make data            # kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip
```

---

## 3. Pipeline Deep Learning & MLOps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull             # dataset complet (~7 Go)
dvc repro            # stage DVC « prepare »
make train evaluate  # entraînement + export ONNX (ou : make pipeline)
```

| Étape | Commande | Sorties |
|---|---|---|
| `prepare` (stage DVC, `dvc.yaml`) | `src/data/prepare.py` | `data/processed/{images,splits.csv,stats.json}` |
| `train` | `src/models/train.py` | `models/best.pt`, `train_metrics.json`, run MLflow |
| `evaluate` | `src/models/evaluate.py` | `model.onnx`, `model_fp16.onnx`, `model_int8.onnx`, `metadata.json`, `test_metrics.json`, version du Model Registry |

L'entraînement se fait sur Colab/Kaggle (GPU) et n'est donc plus un stage DVC. Le **modèle final** est versionné directement avec `dvc add` :
- `models/model.onnx.dvc` : modèle servi par l'API ;
- `models/convnext-focal.pt.dvc` : checkpoint ConvNeXt-Tiny + Focal Loss.

Les checkpoints des autres expériences ne sont pas versionnés.

Pour publier un nouveau modèle :

```bash
dvc add models/model.onnx models/convnext-focal.pt
dvc push                                  # AVANT le git push
git add models/*.dvc models/metadata.json models/*_metrics.json
```

`dvc metrics show` et `dvc params diff` permettent de comparer les versions.

### Stratégie d'entraînement (`params.yaml`)

- **Backbones timm pré-entraînés ImageNet :**
  - `resnet50` sert de baseline ;
  - `efficientnetv2_rw_s` et `convnext_tiny` sont les modèles avancés.
- **Fine-tuning en 2 phases :**
  - Phase 1 (`freeze_epochs`) : la tête seule.
  - Phase 2 : tout le réseau, avec des **LR discriminants** (backbone = `lr × 0.1`), un warmup puis une décroissance cosinus, AdamW et l'AMP.
- **Déséquilibre :** au choix
  - Focal Loss pondérée par classe ;
  - entropie croisée pondérée ;
  - `WeightedRandomSampler`.
  Il ne faut pas cumuler sampler et pondération : ce serait une double correction.
- **Augmentation (train uniquement) :**
  - RandomResizedCrop, flips H/V et rotation 90° (les lésions n'ont pas d'orientation) ;
  - ColorJitter et RandAugment ;
  - MixUp/CutMix (`timm.data.Mixup`), avec des pertes compatibles avec les cibles souples.
- **Early stopping** sur la balanced accuracy de validation.

### Plan d'expériences suggéré (à tracer dans MLflow)

```bash
make baseline                                                   # ResNet50 + CE
python -m src.models.train --set train.loss=weighted_ce         # effet de la pondération
python -m src.models.train --set train.loss=focal               # Focal Loss
python -m src.models.train --set train.mixup_alpha=0 --set train.cutmix_alpha=0   # ablation MixUp/CutMix
python -m src.models.train --set model.name=convnext_tiny       # autre backbone
python -m src.models.train --set train.img_size=224             # effet de la résolution
```

Ces runs fournissent directement le tableau « Baseline vs modèle avancé » du rapport.

### Entraîner sur Kaggle / Colab

```bash
!git clone https://github.com/kaouech-rahim/ham10000-skin-lesion.git && cd ham10000-skin-lesion
!pip install -q -r requirements.txt
# Tracking distant sur DagsHub (MLflow + remote DVC) :
%env MLFLOW_TRACKING_URI=https://dagshub.com/kaouech-rahim/ham10000-skin-lesion.mlflow
%env MLFLOW_TRACKING_USERNAME=<votre_user_dagshub>
!dvc pull && dvc repro
```

Ne mettez jamais le token en clair dans le notebook. Saisissez-le au moment de l'exécution :

```python
import os, getpass
os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("Token DagsHub : ")
```

Sans serveur distant, MLflow écrit dans `./mlruns`. Il suffit alors de télécharger ce dossier et de lancer `mlflow ui`.

### MLflow

Chaque run journalise :
- **les paramètres** : `params.yaml` aplati ;
- **les métriques par époque** : loss, balanced accuracy, F1, rappels par classe, LR ;
- **les artefacts** :
  - les matrices de confusion val et test et le rapport de classification ;
  - la répartition des classes ;
  - le checkpoint ;
  - les modèles ONNX et le benchmark de latence (taille, p50/p95 et accuracy de chaque variante).

Le modèle est enregistré dans le **Model Registry** (`ham10000-classifier`) :
- l'alias `staging` est posé à chaque nouvelle version ;
- l'alias `production` n'est posé que si la balanced accuracy de test bat celle de la version en production. On peut aussi le forcer avec `--promote`.

Les aliases remplacent les « stages », dépréciés depuis MLflow 2.9.

### ONNX & Grad-CAM exportable

ONNX ne contient que le forward. Le Grad-CAM est donc calculé **analytiquement dans le graphe** (`src/utils/gradcam.py`). Pour une tête `pool moyen → [LayerNorm] → Linear`, les poids Grad-CAM se calculent sans autograd :
- tête linéaire : $\alpha_k \propto \partial y_c/\partial p_k = W_{c,k}$ ;
- tête LayerNorm + linéaire (ConvNeXt) : la même dérivée, passée par la LayerNorm (formule en forme fermée).

Le modèle ONNX a donc deux sorties, `logits` et `cam`, et le serveur n'a besoin ni de PyTorch ni de gradients. `tests/test_model.py` vérifie l'**égalité numérique avec Grad-CAM par autograd** pour ResNet, EfficientNet et ConvNeXt.

Variantes exportées :
- **FP32** : écart de logits avec PyTorch vérifié, de l'ordre de 1e-7 ;
- **FP16** : pour GPU ;
- **INT8** : quantification statique QDQ par canal, calibrée sur la validation. Elle ne porte que sur Conv/Gemm : la branche Grad-CAM reste en FP32.

---

## 4. API FastAPI

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | état, GPU, version et variante du modèle |
| GET | `/model/metadata` | architecture, classes, métriques val/test, benchmark ONNX |
| POST | `/predict` | `UploadFile` → probabilités des 7 classes (triées), alerte de risque, latences |
| POST | `/predict/visualize` | PNG annoté par Grad-CAM (`?format=png`, par défaut) ou JSON avec image, heatmap et overlay en base64 (`?format=json&alpha=0.45`) |
| GET | `/monitoring/stats` | latence moyenne/p95, confiance moyenne, part des prédictions peu confiantes, distribution des classes prédites (signal de dérive) |

Points de conception :
- les réponses sont validées par des **schémas Pydantic stricts** (`extra="forbid"`) ;
- l'inférence s'exécute dans un threadpool pour ne pas bloquer la boucle asyncio ;
- le prétraitement reproduit exactement celui de la validation : resize sur le côté court, center-crop, normalisation ImageNet.

Codes d'erreur :

| Code | Cas |
|---|---|
| `400` | image invalide ou trop petite, mauvais type MIME, fichier vide |
| `413` | fichier de plus de 10 Mo |
| `422` | champ `file` manquant |
| `503` | modèle non chargé |

```bash
curl -F file=@lesion.jpg http://localhost:8000/predict
curl -F file=@lesion.jpg "http://localhost:8000/predict/visualize" -o gradcam.png
```

---

## 5. Frontend React

Stack : Vite, React 18 et Tailwind CSS v4.
- **3 modes d'entrée :**
  - glisser-déposer ;
  - galerie d'exemples, avec affichage de la vérité terrain ;
  - **webcam**, en capture unique ou en **inférence continue**.
- **Résultats :**
  - **barres de probabilités animées**, rouges pour les classes malignes ;
  - bandeau d'alerte si la classe prédite est maligne ;
  - **visualiseur Grad-CAM** avec curseur d'opacité et mode comparaison.
- **Latence :** client (aller-retour), serveur total, inférence ONNX et prétraitement, avec un historique des dernières requêtes.

```bash
cd frontend && npm install && npm run dev     # http://localhost:5173 (proxy /api -> :8000)
```

**Images de la galerie.** `frontend/public/samples/` contient **14 images du jeu de TEST HAM10000** (2 par classe), commitées dans Git pour que la galerie fonctionne dès le clone. Elles ne servent jamais à l'entraînement. Elles sont régénérées par `make samples` (`python -m src.data.export_samples --per-class 2`).

Ces images sont distribuées sous licence **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** (usage non commercial, attribution obligatoire). Source : Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Sci. Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

---

## 6. Tests & CI/CD

```bash
make test     # pytest tests/ (ML) + backend/tests (API)
make lint     # black, flake8, eslint
```

- `.github/workflows/ci-tests.yml` :
  - black et flake8 ;
  - pytest pour le backend (mini-modèle ONNX réel construit dans les fixtures, pas de poids nécessaires) ;
  - pytest pour le ML (split sans fuite, Focal Loss, Grad-CAM exportable égal à l'autograd) ;
  - ESLint et build Vite.
- `.github/workflows/docker-build.yml` :
  - build des images multi-stage backend et frontend, avec cache GHA ;
  - push sur GHCR depuis `main` ou un tag ;
  - smoke test `docker compose` : l'API démarre, `/health` répond et `/predict` renvoie bien 503 sans modèle.

---

## 7. Arborescence

```
.github/workflows/   ci-tests.yml, docker-build.yml
data/                raw/ + processed/   (DVC)
src/
  config.py          classes, constantes, chargement params
  data/              prepare.py (split par lésion), dataset.py, export_samples.py
  models/            architecture.py, losses.py, train.py, evaluate.py
  utils/             gradcam.py, metrics.py
tests/               tests du code ML
backend/             FastAPI : app/{main,schemas,inference,visualization,config}.py, tests/, Dockerfile
frontend/            React : src/{components,services/api.js,App.jsx}, Dockerfile, nginx.conf
models/              model.onnx.dvc, convnext-focal.pt.dvc (DVC) ; metadata.json, *_metrics.json (Git)
dvc.yaml (stage prepare), params.yaml, docker-compose.yml, Makefile, .gitattributes
```

## 8. Répartition des rôles (trinôme)

| Rôle | Responsabilités principales |
|---|---|
| Lead Deep Learning | `src/data`, `src/models`, ablations, Grad-CAM, quantification, analyse des erreurs |
| Lead MLOps / DevOps | DVC + remote, MLflow + Registry, Dockerfiles, docker-compose, GitHub Actions, monitoring |
| Lead Full-Stack | `backend/` (API, schémas, tests), `frontend/` (UX, webcam, Grad-CAM, latence) |

## 9. Checklist soutenance (démo live)

- [ ] `docker compose up --build` sur une machine propre
- [ ] Inférence par upload, galerie et webcam, avec Grad-CAM et latences affichées
- [ ] Swagger `/docs` : montrer un 400 (fichier texte) et le schéma Pydantic
- [ ] MLflow : comparaison baseline vs modèle avancé, artefacts, alias `production`
- [ ] DVC : `dvc dag`, `dvc metrics show`, `dvc params diff`
- [ ] GitHub Actions au vert
- [ ] Savoir expliquer :
  - le split par `lesion_id` ;
  - le choix de la balanced accuracy ;
  - la Focal Loss ;
  - le Grad-CAM analytique ;
  - le compromis FP32/INT8.

## Références

- Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset*. Sci. Data 5, 180161 (2018).
- Codella N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018 (ISIC)*. arXiv:1902.03368.
- Selvaraju R. et al. *Grad-CAM*. ICCV 2017. Lin T.-Y. et al. *Focal Loss*. ICCV 2017.
