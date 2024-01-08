
# Rapport PRD

# Segmentation 3D d'IRM de cerveaux animal

> Léo Boulanger | Polytech Tours | 2023-2024


## Introduction
L'anatomie est depuis longtemps une science étudiée par les Humain, principalement pour comprendre le fonctionnement notre propre corps et ceux d'autrui. Parmi les différentes structures, l'organe le plus complexe pour une grande majorité des êtres vivants est le cerveau. Cependant, ce n'est qu'à partir du milieu du XXème siècle que

### Acteurs
Ce projet recherche et développement est en lien avec le projet du doctorant Antoine Bourlier, visant à ...

### Objectif
L'objectif de ce projet est de développer un outil en Python permettant à des utilisateurs inexpérimentés en informatique d'obtenir une segmentation d'un cerveau à partir de ses données d'IRM au format NIfTI.

### Hypothèses
Il existe plusieurs types de méthodes pour segmenter un volume 3D, notamment pour le cerveau, qui est cible de nombreuses recherches. Pour ce projet, une méthode segmentation doit être sélectionnée.

Les méthodes ***basées intensité*** sont parmi les plus simples. Le principe est d'utiliser la valeur des mesures pour les séparer selon leurs intensités.

Les méthodes ***basées atlas*** impliquent l'utilisation d'une liste de références, où chaque objet de cette liste correspond à une ou un ensemble de caractéristiques d'une structure recherchée. Ces méthodes sont donc plus complexes que celles par intensité car elles nécessitent d'avoir déjà analysé et décrit précisément le sujet en amont.

Enfin, les méthodes ***basées apprentissage supervisé***, sont populaires depuis plusieurs années. Elles utilisent une base d'apprentissage pour entraîner un modèle, qui est ensuite utilisé pour prédire des résultats

Au moment de ce projet, peu de segmentations labélisées de cerveaux d'animaux sont disponibles, donc la méthode basée intensité a été sélectionnée.  
Plus de détails à propos de ces méthodes sont décrites dans la partie "## État de l'art"





Il existe plusieurs types de méthodes pour segmenter un volume 3D, notamment pour le cerveau, qui est cible de nombreuses recherches. Pour ce projet, une méthode segmentation doit être sélectionné	e.

Si une base d’apprentissage est fournie pour ce projet, il sera possible d’implémenter la méthode AssemblyNet. Le modèle d’AssemblyNet utilise une grande base d’images pour être efficace :
•	Le dataset d’entraînement : 45 IRM T1w complètement labélisés manuellement
•	Un dataset de tests : 19 IRM T1w complètement et manuellement labélisés
•	Un dataset de 8 IRM T1w labélisés manuellement selon le protocole BrainCOLOR utilisé pour l’expérimentation scan-rescan.
•	Un pathological dataset de 29 IRM T1w labélisés manuellement selon le protocole BrainCOLOR
•	Un lifespan dataset de 360 IRM T1w non labélisés
Cependant, si l’implémentation d’AssemblyNet pour les animaux est trop complexe et nécessite d’y accorder trop de temps, la méthode [*** NOM D’UNE AUTRE MÉTHODE ? une autre méthode de deep learning, plus simple ***] pourra être utilisée.

Au moment de ce projet, peu de segmentations labélisées de cerveaux d'animaux sont disponibles, donc une méthode basée intensité est préférablement sélectionnée. Dans ce cas, une solution est d’appliquer un clustering suivant la méthode fuzzy c-means permettant d’obtenir un certain nombre de clusters, et leurs centres, sur lesquels appliquer un algorithme de segmentation par intensité local, comme le region growing 3D. Cette méthode fuzzy c-means est toutefois sensible au bruit, et de haute complexité, mais elle ne nécessite pas d’interactivité utilisateur, est paramétrique, et ne nécessite pas de connaissance a priori. Les recherches récentes mettent en avant la méthode 3D unsupervised modified spatial fuzzy c-means, qui [***]. 






### Bases méthodologiques



## État de l'art
Cette partie présente une synthèse des travaux et recherches publiées avant 2023.

### Reprise de l'existant
Pour ce projet, deux algorithmes ont déjà été écrits par Antoine bourlier: l'un pour prétraiter les données des mesures d'une IRM au format NIfTI, et l'autre effectue une segmentation split and merge, basée intensité.

### Méthodes de segmentation basées intensité


### Méthodes de segmentation basées atlas


### Méthodes de segmentation basées apprentissage supervisé


### Autres méthodes de segmentation



## Analyse et conception



## Bilan et conclusion



## Bibliographie
> [fuzzy-c-means: An implementation of Fuzzy C-means clustering algorithm - Dias, 2019](https://git.io/fuzzy-c-means)  
> [Automated 3D region growing algorithm based on an assessment function - Revol-Muller, 2002](https://www.sciencedirect.com/science/article/pii/S0167865501001167)


## Annexe