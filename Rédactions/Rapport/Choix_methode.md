# Choix de la méthode de segmentation

## Références:
[3D Unsupervised modified spatial fuzzy c-mean](https://link.springer.com/article/10.1007/s10044-019-00806-2)
[Tutoriel Python region growing + clustering segmentation](https://sbme-tutorials.github.io/2019/cv/notes/6_week6.html)


### Clustering + Intensité :
- Les segmentations basées "fuzzy" sont les plus adaptée pour segmenter le cerveau car elle évite l'impact de bias field et du bruit sur l'analyse.
- La méthode fuzzy c-mean est simple et efficace pour segmenter les IRM, mais elle risque de produire des résultats imprécis en présence de "bias field" et de déformations.
- Elle peut toutefois devenir performante contre le bruit et l'inhomogénéité des intensités en incluant les informations spatiales.




### Code:
https://github.com/omadson/fuzzy-c-means/blob/master/fcmeans/main.py