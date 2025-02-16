# README - Projet YOLOv8s pour la Détection et Estimation de Pose des Macaques

## 📬 Contact
Ce projet a été réalisé par **Nermine Bacha** et **Cyrine Zribi**.

---

## 1️⃣ Introduction
Ce projet vise à **entraîner un modèle YOLOv8s** sur le dataset **MacaquePose** pour détecter les macaques et estimer leurs poses à partir d’images.

L’objectif principal est d’évaluer les performances d’un modèle avancé de détection et d’estimation de pose sur un dataset spécifique et d’analyser en détail ses forces et ses limites. Nous allons dans un premier temps **détecter la présence des macaques dans une boîte englobante**, puis analyser **la position exacte de leurs membres à travers l'estimation de pose**.

Ce type d'approche est particulièrement utile dans **l'étude comportementale des animaux**, permettant par exemple aux biologistes et aux chercheurs de suivre et d'analyser les mouvements des macaques en milieu naturel ou en captivité. L'analyse de pose pourrait également être appliquée pour **détecter des anomalies motrices** ou pour **comparer le comportement des primates dans différentes conditions**.

Nous avons choisi de travailler sur ce sujet car **YOLOv8s est largement utilisé pour la détection et l'estimation de pose des humains**, mais bien que des datasets existent pour les animaux, **très peu de modèles sont testés et optimisés pour eux**. Nous avons donc voulu **expérimenter l'application de YOLOv8s sur un dataset animalier et évaluer ses performances**.

---

## 2️⃣ Présentation du Dataset : MacaquePose

Le **dataset MacaquePose** a été conçu pour la **détection d'objets et l'estimation de pose** des macaques dans divers environnements (zoo, nature). Il contient :

- **13 000 images** annotées avec des **boîtes englobantes (bounding boxes)** autour des macaques.
- **Annotations des points clés du squelette** pour l'estimation de pose (tête, pattes, queue, etc.).
- **Variabilité des poses et des contextes** pour assurer une meilleure généralisation du modèle.

Nous avons choisi ce dataset car il représente un défi intéressant pour l’**apprentissage automatique appliqué à la reconnaissance d’animaux sauvages**, avec des conditions d’éclairage et d’arrière-plan variées.

📌 **Lien vers le dataset :** [MacaquePose](http://www2.ehub.kyoto-u.ac.jp/datasets/macaquepose/)

---

## 3️⃣ Prétraitement et Entraînement du Modèle

### 📌 **Technologies utilisées**
- **YOLOv8s** (modèle optimisé pour la détection d’objets et l’estimation de pose).
- **Python, PyTorch, OpenCV, Seaborn, Pandas** pour la gestion des données et la visualisation.
- **Google Colab avec GPU** pour accélérer l’entraînement.

### 🛠 **Préparation des données**
Nous avons commencé par **télécharger et explorer** les données pour comprendre la répartition des annotations :
- Distribution des labels et des positions des macaques.
- Vérification des tailles d’images et des annotations.
- Augmentations des données : 
  - Flip horizontal : 50%
  - Variation de l’échelle : 50%
  - Auto-augmentation : `randaugment`
  - Suppression aléatoire de parties d’image (`erasing`) : 40%

### 🔥 **Paramètres d'entraînement**
```yaml
model: yolov8s-pose.pt
data: /content/config.yaml
epochs: 3
batch: 16
imgsz: 640
optimizer: auto
confidence threshold: 0.7
```
Le modèle a été entraîné en **3 époques**, ce qui est relativement court, mais a permis d’obtenir de bons résultats.

---

## 4️⃣ Résultats et Analyse des Performances

### 📊 **Métriques Clés**
| Métrique | Score |
|----------|--------|
| **mAP@0.5 (bounding box)** | **0.961** |
| **mAP@0.5 (pose estimation)** | **0.839** |
| **F1-Score optimal** | **0.94 à 0.536 confiance** |

### 📈 **Visualisation et Interprétation des Résultats**

#### 1️⃣ **Courbes de Précision-Rappel (PR) et F1-score**
Les courbes PR montrent que le modèle atteint une **précision proche de 1.0** pour un **rappel inférieur à 0.9**, puis chute fortement.
- Pour la **détection des boîtes**, **mAP@0.5 = 0.961**, ce qui signifie que le modèle fait très peu d’erreurs de classification.
- Pour **l’estimation de pose**, **mAP@0.5 = 0.839**, légèrement plus faible, ce qui s’explique par la complexité du problème.

#### 2️⃣ **Matrice de Confusion**
- **1567 prédictions correctes sur 1886 cas**.
- Quelques erreurs dues à des **poses inhabituelles** ou à des **occlusions partielles**.
- **Interprétation** : Un excellent taux de reconnaissance des macaques, mais une sensibilité aux **scènes complexes**.

#### 3️⃣ **Distribution des Labels**
- Les annotations sont bien réparties, ce qui montre que le dataset est **équilibré en termes de position et taille des objets**.
- Les largeurs et hauteurs des boîtes suivent une distribution **logiquement proportionnelle**, confirmant une annotation correcte.

#### 4️⃣ **Visualisation des Prédictions**
Nous avons superposé les prédictions du modèle sur les images d’évaluation :
- **Les boîtes englobantes sont bien alignées avec les objets**.
- **Les squelettes pour l’estimation de pose sont bien détectés**.
- Quelques erreurs d’alignement sur **les postures inhabituelles et les images de faible qualité**.

---

## 5️⃣ Conclusion

Ce projet a permis d’évaluer les performances du modèle YOLOv8s sur un dataset dédié aux animaux. Nous avons constaté que le modèle obtenait des résultats très satisfaisants en détection de macaques avec une mAP@0.5 de 0.961. L’estimation de pose est présente avec mAP@0.5 de 0.839, montrant que certains ajustements pourraient améliorer encore plus la précision mais reste très valable.

L’expérience a également mis en évidence l’importance d’un dataset équilibré et bien annoté pour entraîner des modèles efficaces. En ajustant l’augmentation des données, les hyperparamètres, ou en ajoutant plus d’époques d’entraînement, on pourrait encore améliorer ces résultats.

Enfin, ce projet ouvre des perspectives pour des applications futures, notamment dans l’étude comportementale des primates, la conservation des espèces, et même l’adaptation des modèles de détection et d’estimation de pose à d’autres espèces animales.
