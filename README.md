# EDP Parabolique de la chaleur : Simulation et Visualisation

---

Bienvenue sur la page Github consacrée à la simulation des solutions du problème considéré dans l'article associé.

- discontinuous case.py : Calcul des solutions approchées (pour le choix $u_0$ discontinu) du schéma et calcul de l'énergie.
- heat_equation.py : Script pour calculer les solution approchées (pour le choix $u_0$ continu) du schéma et les comparer avec les solutions exactes + calcul et affichage de l'énergie en fonction du temps 
- heat_errors.py : Script utilisé pour tracer les courbes d'erreur entre le schéma et la solution exacte (/!\ attention, le script prend du temps à tourner)
- infinite_speed_propagation.py : Illustration du phénomène de propagation à vitesse infinie à l'aide d'un même graphique
- propagation_anim.py : Animation matplotlib pour visualiser le phénomène de propagation à vitesse infinie (c'est bonus)
