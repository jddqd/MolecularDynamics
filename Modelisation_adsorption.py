
"""
Created on Tue April  12 20:39:40 2022

@author: c.malerba / g.pradel
"""


#=================================================
# Programme python adsorption
#=================================================





##################################################
# Modules
##################################################

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rd



##################################################
# Fonction distance
##################################################

def distance(dx,dy,dz):
    return np.sqrt(dx**2 + dy**2 + dz**2)



##################################################
# Fonction répartition molécule
##################################################

def nombre_molecule_initial(N,r):
    nb_site_libre = N/(r + 1)
    nb_chlore = N - nb_site_libre
    return nb_site_libre, nb_chlore



##################################################
# Nombre de molécules à chaque instant
##################################################

def nombre_molécules(etats):
    sl = 0 # site libre
    so = 0 # site occupé
    cl = 0 # chlore
    ca = 0 # chlore adsorbé
    for e in etats:
        if e == 0:
            sl += 1
        if e == 1:
            so += 1
        if e == 2:
            cl += 1
        if e == 3:
            ca += 1
    return sl, so, cl, ca



##################################################
# Initialisation
##################################################

def init(N,V,D,E,F,rayons,eps):
    
    # tableaux initiaux
    
    x = np.zeros(N) # positions
    y = np.zeros(N)
    z = np.zeros(N)
    
    vx = np.zeros(N) # vitesses
    vy = np.zeros(N)
    vz = np.zeros(N)
    
    radius = np.zeros(N) # rayons
    etats = np.zeros(N) # etats des particules
    ind_ads = [] # indices des molécules adsorbées
    
    nb_s, nb_c = nombre_molecule_initial(N,r) # répartition
    nb_site_libre = round(nb_s) # nombre de molécules de charbon
    
    print ('Initialisation')
    
    for i in range(N):
        
        print (i+1,' /',N)
        
        if i == 0: # initialisation de la première molécule
            if i <= nb_site_libre:
                radius[i] = rayons[0]
                etats[i] = 0
            if i > nb_site_libre:
                radius[i] = rayons[2]
                etats[i] = 2
            x[i] = rd.uniform(radius[i], D - radius[i])
            y[i] = rd.uniform(radius[i], E - radius[i])
            z[i] = rd.uniform(radius[i], F - radius[i])
        
        if i != 0: # génération d'une particule aléatoire
            essai = False
            while essai == False:
                essai = True
                e_try = 0
                r_try = rayons[0]
                if i > nb_site_libre:
                    e_try = 2
                    r_try = rayons[2]
                x_try = rd.uniform(r_try, D - r_try)
                y_try = rd.uniform(r_try, E - r_try)
                z_try = rd.uniform(r_try, F - r_try)
                
                # test si la particule aléatoire fonctionne
                
                for j in range(0,i): 
                    xj = x[j] 
                    yj = y[j] 
                    zj = z[j]
                    rj = radius[j]
                    d = distance(xj - x_try, yj - y_try, zj - z_try)
                    if d < r_try + rj:
                        essai = False
            
            # ajout de la particule aléatoire qui fonctionne
            
            radius[i] = r_try
            etats[i] = e_try
            x[i] = x_try
            y[i] = y_try
            z[i] = z_try
        
        # génération des vitesses initiales
        
        phi = rd.uniform(0, 2*np.pi)
        theta = np.arccos(1 - 2*rd.uniform(0, 1))
        vx[i] = V*np.sin(theta)*np.cos(phi)
        vy[i] = V*np.sin(theta)*np.sin(phi)
        vz[i] = V*np.cos(theta)
    
    return x, y, z, vx, vy, vz, radius, etats, ind_ads



##################################################
# Création du dictionnaire
##################################################

# Les clés sont des tuples coordonnées de la case 
# Les valeurs sont la liste des particules présentes

def dico_cellules(N,x,y,z,d_max):
    A = {}
    for i in range(N):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ix = int(xi/d_max)
        iy = int(yi/d_max)
        iz = int(zi/d_max)
        if (ix, iy, iz) in A:
            A[(ix, iy, iz)].append(i)
        else:
            A[(ix, iy, iz)] = [i]
    return A



##################################################
# Calcul du vecteur accélération
##################################################

def acceleration_toutes_particules(vx,vy,vz,g,h,gamma):
    ax = -h*vx - gamma*vx
    ay = -h*vy - gamma*vy
    az = -h*vz - gamma*vz
    return ax, ay, az



##################################################
# Calcul de l'énergie
##################################################

def energie(z,vx,vy,vz,g):
    Ec = 0.5*np.sum(vx**2 + vy**2 + vz**2)
    Ep = g*np.sum(z)
    Ep = 0
    return Ec, Ep, Ec+Ep



##################################################
# Calcul des chocs entre les particules
##################################################

def chocs_particules(N,x,y,z,vx,vy,vz,radius,etats,ind_ads,eps,d_max):
    
    cellules = dico_cellules(N, x, y, z, d_max) # cases contenant des particules
    liste_chocs = [] # liste qui contient les particules ayant déjà eu un choc pour ne pas les faires se cogner deux fois
    
    for i in range(N):
        if etats[i] == 3:
            liste_chocs.append(i) # le chlore adsorbé n'intervient pas
    
    for i in range (N):
        if i not in liste_chocs:
            xi = x[i]
            yi = y[i]
            zi = z[i]
            vxi = vx[i]
            vyi = vy[i]
            vzi = vz[i]
            ri = radius[i]
            ei = etats[i]
            ix = int(xi/d_max)
            iy = int(yi/d_max)
            iz = int(zi/d_max)
            V = [] # liste des voisins susceptible de cogner
            M = [-1,0,1]
            for k in M:
                for p in M:
                    for q in M:
                        if (ix + k, iy + p, iz + q) in cellules:
                            voisins = cellules[(ix + k, iy + p, iz + q)]
                            for v in voisins:
                                if v != i:
                                    V.append(v)
            for j in V: 
                xj = x[j]
                yj = y[j]
                zj = z[j]
                vxj = vx[j]
                vyj = vy[j]
                vzj = vz[j]
                rj = radius[j]
                ej = etats[j]
                r = distance(xi - xj, yi - yj, zi - zj)
                d = ri + rj
                xm = (xi + xj)/2
                ym = (yi + yj)/2
                zm = (zi + zj)/2
                
                if r < d: # une collision se produit
                    
                    # Choc
                    
                    if (ei,ej) in L_choc:
                        
                        print('Choc')
                        
                        # vecteur unitaire allant de i à j
                        
                        nx = (xj - xi)/r
                        ny = (yj - yi)/r
                        nz = (zj - zi)/r
                        vjvi_dot_n = (vxj - vxi)*nx + (vyj - vyi)*ny + (vzj - vzi)*nz
                        
                        vx[i] += vjvi_dot_n*nx
                        vy[i] += vjvi_dot_n*ny
                        vz[i] += vjvi_dot_n*nz
                        vx[j] -= vjvi_dot_n*nx
                        vy[j] -= vjvi_dot_n*ny
                        vz[j] -= vjvi_dot_n*nz
                        
                        x[i] = xm - (d + eps)*nx/2
                        y[i] = ym - (d + eps)*ny/2
                        z[i] = zm - (d + eps)*nz/2
                        
                        x[j] = xm + (d + eps)*nx/2
                        y[j] = ym + (d + eps)*ny/2
                        z[j] = zm + (d + eps)*nz/2
                        
                        liste_chocs.append(i)
                        liste_chocs.append(j)
                    
                    # Adsorption
                    
                    if (ei,ej) in L_adso:
                        
                        tmp = rd.random()
                        if tmp <= proba_ad: # test proba adsorption
                            
                            print('Adsorption')
                            if ei == 0: # particule i = site libre
                                
                                etats[i] = 1
                                etats[j] = 3
                                ind_ads.append(j)
                                
                                nx = (xj - xi)/r
                                ny = (yj - yi)/r
                                nz = (zj - zi)/r
                                vjvi_dot_n = (vxj - vxi)*nx + (vyj - vyi)*ny + (vzj - vzi)*nz
                                
                                vx[i] += vjvi_dot_n*nx
                                vy[i] += vjvi_dot_n*ny
                                vz[i] += vjvi_dot_n*nz
                                vx[j] = 0
                                vy[j] = 0
                                vz[j] = 0
                                
                                x[i] = xm - (d + eps)*nx/2
                                y[i] = ym - (d + eps)*ny/2
                                z[i] = zm - (d + eps)*nz/2
                                
                                x[j] = 0
                                y[j] = 0
                                z[j] = 0
                                
                                liste_chocs.append(i)
                                liste_chocs.append(j)
                            
                            if ei == 2: # particule i = chlore
                                
                                etats[i] = 3
                                etats[j] = 1
                                    
                                ind_ads.append(j)
                                
                                nx = (xj - xi)/r
                                ny = (yj - yi)/r
                                nz = (zj - zi)/r
                                vjvi_dot_n = (vxj - vxi)*nx + (vyj - vyi)*ny + (vzj - vzi)*nz
                                
                                vx[i] = 0
                                vy[i] = 0
                                vz[i] = 0
                                vx[j] -= vjvi_dot_n*nx
                                vy[j] -= vjvi_dot_n*ny
                                vz[j] -= vjvi_dot_n*nz
                                
                                x[i] = 0
                                y[i] = 0
                                z[i] = 0
                                
                                x[j] = xm + (d + eps)*nx/2
                                y[j] = ym + (d + eps)*ny/2
                                z[j] = zm + (d + eps)*nz/2
                                
                                liste_chocs.append(i)
                                liste_chocs.append(j)
                        
                        else:
                            
                            print('Adsorption ratée')
                            nx = (xj - xi)/r
                            ny = (yj - yi)/r
                            nz = (zj - zi)/r
                            vjvi_dot_n = (vxj - vxi)*nx + (vyj - vyi)*ny + (vzj - vzi)*nz
                            
                            vx[i] += vjvi_dot_n*nx
                            vy[i] += vjvi_dot_n*ny
                            vz[i] += vjvi_dot_n*nz
                            vx[j] -= vjvi_dot_n*nx
                            vy[j] -= vjvi_dot_n*ny
                            vz[j] -= vjvi_dot_n*nz
                            
                            x[i] = xm - (d + eps)*nx/2
                            y[i] = ym - (d + eps)*ny/2
                            z[i] = zm - (d + eps)*nz/2
                            
                            x[j] = xm + (d + eps)*nx/2
                            y[j] = ym + (d + eps)*ny/2
                            z[j] = zm + (d + eps)*nz/2
                            
                            liste_chocs.append(i)
                            liste_chocs.append(j)
    
    return x, y, z, vx, vy, vz



##################################################
# Calcul des chocs avec les paroies
##################################################

def chocs_parois(N,x,y,z,vx,vy,vz,radius,etats,D,E,F,V,plafond,Thermostat,eps):
    for i in range(N):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        ri = radius[i]
        ei = etats[i]
        
        if ei != 3:
            if xi > D - ri :
                vx[i] *= -1 # choc élastique
                x[i] = D - ri - eps
            if xi < ri :
                vx[i] *= -1 # choc élastique
                x[i] = ri + eps
            
            if yi > E - ri:
                vy[i] *= -1
                y[i] = E - ri - eps
            if yi < ri :
                vy[i] *= -1 # choc élastique
                y[i] = ri + eps
            
            if plafond == True: # choc élastique sur le plafond
                if zi > F - ri:
                    vz[i] *= -1
                    z[i] = F - ri - eps
            if zi < ri:
                vz[i] *= -1
                z[i] = ri + eps
    
    return x, y, z, vx, vy, vz



##################################################
# Prise en compte de la désorption
##################################################

def desorption(N,x,y,z,vx,vy,vz,radius,etats,ind_ads,eps,d_max):
    for i in range(N):
        ei = etats[i]
        if ei == 1: # site occupé
            tmp = rd.random()
            # print(tmp)
            if tmp <= proba_des: # test proba désorption
                print('Désorption')
                j = ind_ads.pop()
                etats[i] = 0
                etats[j] = 2
                x[j] = x[i] + d_max
                y[j] = y[i] + d_max
                z[j] = z[i] + d_max
                vx[j] = -vx[i]
                vy[j] = -vy[i]
                vz[j] = -vz[i]



##################################################
# Affichage graphique
##################################################

def affiche(N,x,y,z,vx,vy,vz,D,E,F,radius,etats,compteur,dossier,d_images,t,SL,SO,CL,CA,MOL,T,Thermostat,g,Temp_eq,plafond,h):
    file = dossier+'/'+str(compteur).zfill(d_images)+'.png'
    fig = plt.figure(compteur)
    plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10, wspace=0.25, hspace=0.25)
    plt.gcf().set_size_inches(12, 12*2/3)
    
    # Trajectoires
    
    ax = fig.add_subplot(121, projection='3d')
    
    ax.set_xlim(0,D)
    ax.set_ylim(0,E)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    
    if plafond == True:
        ax.set_zlim(0,F)
    else:
     	H = Temp_eq/g
     	ax.set_zlim(0,3*H)
    
    for i in range(N):
        if etats[i] == 0:
            ax.scatter(x[i],y[i],z[i],color='orangered')
        if etats[i] == 1:
            ax.scatter(x[i],y[i],z[i],color='black')
        if etats[i] == 2:
            ax.scatter(x[i],y[i],z[i],color='limegreen')
        # if etats[i] == 3:
        #     ax.scatter(x[i],y[i],z[i],color='white')
    
    # Rapport des particules
    
    plt.subplot(122)
    plt.grid()
    
    plt.xlabel('t')
    plt.ylabel('nombre de particules')
    
    plt.plot(T,MOL,label='Nombre total',linewidth=2,color='blue')
    plt.plot(T,CL,label='Chlore',linewidth=2,color='limegreen')
    plt.plot(T,SL,label='Site libre',linewidth=2,color='orangered')
    plt.plot(T,SO,label='Site occupé',linewidth=2,color='black')
    
    plt.legend(loc='lower left')
    
    # Sauvegarde de chaque image 
    
    plt.savefig(file)
    plt.close(compteur)



##################################################
# Fonction à appeler pour lancer le programme
##################################################

def main(N,rayons,g,T_init,h,D,E,F,plafond,Thermostat,Q,tmax,tol_deplacement,images,N_images): 
    
    V = np.sqrt(3*T_init) # norme des vitesses initiales
    
    eps = min(rayons)/10000 # sert à décaler légèrement les particules lors des chocs pour éviter qu'elles restent coincées
    
    tol_deplacement *= min(rayons) # valeur maximale de déplacement autorisé
    
    gamma = 0 # pour le thermostat éventuel de Nose-Hover
    
    x, y, z, vx, vy, vz, radius, etats, ind_ads = init(N, V, D, E, F, rayons, eps) # création des conditions initiales
    d_max = 2*max(rayons) # taille des cellules pour la recherche des voisins
    
    # Stockage de l'historique des valeurs pour l'affichage graphique
    
    sl,so,cl,ca = nombre_molécules(etats)
    SL = [sl]
    SO = [so]
    CL = [cl]
    CA = [ca]
    MOL = [N]
    Ec, Ep, Et = energie(z, vx, vy, vz, g)
    
    # Calcul de la température théorique d'équilibre
    # Si g = 0, alors la température ne doit pas changer,
    # Si g != 0, alors à l'équilibre on a :
    # Ep = N*k_B*T et Ec = (3/2)*N*k_B*T (car 3 ddl)
    # Donc T_eq = 2*Et/(5*k_B*N) et on prend k_B = 1 pour normaliser
    
    if Thermostat == True:
        Temp_equilibre = T_init
    else:
        if g != 0 :
            Temp_equilibre = 2*Et/(5*N)
        else:
            Temp_equilibre = Et/N
    
    t = 0
    T = [t] # stockage des dates successives d'intégration
    
    compteur = 0
    t_compteur = tmax/N_images
    
    # Création du dossier de stockage des images et du fichier parametres.txt
    
    if images == True:
        d_images = len(str(N_images)) # nombre de décimales pour la sauvegarde des images
        date = datetime.datetime.now()
        dossier = date.strftime('%Y-%m-%d-%H-%M-%S')
        os.mkdir(dossier)
        file_data = open(dossier+'/'+'parametres.txt','w')
        
        file_data.write(dossier+"\n")
        file_data.write('\n')
        file_data.write('Variables physiques : \n')
        file_data.write('\n')
        file_data.write('Température initiale : T_init = %s \n' % T_init)
        file_data.write('Coefficient de frottement : h = %s \n' % h)
        file_data.write('Constante de gravitation : g = %s \n' % g)
        file_data.write('Largeur de l\'enceinte : D = %s \n' % D)
        file_data.write('Hauteur de l\'enceinte : E = %s \n' % E)
        file_data.write('Profondeur de l\'enceinte : F = %s \n' % F)
        file_data.write('Masse du thermostat de Nose-Hover : Q = %s \n' % Q)
        file_data.write('Rayon d\'un atome de chlore : r_chlore = %s \n' % r_chlore)
        file_data.write('Rayon d\'un site libre : r_site_libre = %s \n' % r_site_libre)
        file_data.write('Rayon d\'un site occupé : r_site_occupé = %s \n' % r_site_occupé)
        file_data.write('Probabilité d\'adsorption : proba_ad = %s \n' % proba_ad)
        file_data.write('Probabilité de désorption : proba_des = %s \n' % proba_des)
        file_data.write('Rayons des particules : rayons = %s \n' % rayons)
        file_data.write('Rapport des différentes particules : r = %s \n' % r)
        file_data.write('\n')
        file_data.write('Paramètres de la simulation : \n')
        file_data.write('\n')
        file_data.write('Présence d\'un plafond : plafond = %s \n' % plafond)
        file_data.write('Présence d\'un thermostat : Thermostat = %s \n' % Thermostat)
        file_data.write('Nombre de molécules : N = %s \n'% N)
        file_data.write('Temps de la simulation : t_tot = %s \n' % tmax)
        file_data.write('Pas de temps : tol_deplacement = %s \n' % tol_deplacement)
        file_data.write('Nombre d\'images N_images = %s \n' % N_images)
        file_data.write('Stockage d\'images : images = %s \n' % images)
        
        file_data.close()
    
    # Boucle de calcul
    
    while t < tmax: # on boucle tant que le temps max de simul n'est pas atteint
    
        # Procédure Verlet avec pas de temps adaptatif
        
        ax1, ay1, az1 = acceleration_toutes_particules(vx, vy, vz, g, h, gamma)
        
        dt = tol_deplacement/np.max(np.sqrt(vx**2+vy**2+vz**2)) # aucune particule ne doit avancer de plus de tol_deplacement
        
        x += vx*dt + ax1*dt**2
        y += vy*dt + ay1*dt**2
        z += vz*dt + az1*dt**2
        
        ax2, ay2, az2 = acceleration_toutes_particules(vx, vy, vz, g, h, gamma)
        vx += (ax1 + ax2)*dt/2
        vy += (ay1 + ay2)*dt/2
        vz += (az1 + az2)*dt/2
        
        # Thermostat de Nose-Hover si Thermostat = True
        
        if Thermostat == True:
            Ec, Ep, Et = energie(z, vx, vy, vz, g)
            Temp = 2*Ec/(3*N)
            gamma += -2/Q*(T_init - Temp)*dt
        
        # Prise en compte des parois
        
        x, y, z, vx, vy, vz = chocs_parois(N, x, y, z, vx, vy, vz, radius, etats, D, E, F, V, plafond, Thermostat, eps)
        
        # Chocs inter-particules
        
        x, y, z, vx, vy, vz = chocs_particules(N, x, y, z, vx, vy, vz, radius, etats, ind_ads, eps, d_max)
        
        # Incrémentation du temps
        
        t += dt
        
        # Ici on teste si on doit stocker les données et procéder à l'affichage graphique
        
        if t > compteur*t_compteur:
            compteur += 1
            print(compteur)
            # print('il y a',len(ind_ads),'adsorbé')
            
            # prise en compte de la désorption
            
            desorption(N, x, y, z, vx, vy, vz, radius, etats, ind_ads, eps, d_max)
            
            # Enregistrement des variables : Nombre de chaque type de particules
            
            sl, so, cl, ca = nombre_molécules(etats)
            SL.append(sl)
            SO.append(so)
            CL.append(cl)
            CA.append(ca)
            T.append(t)
            MOL.append(N)
            
            if images == True:
                affiche(N, x, y, z, vx, vy, vz, D, E, F, radius, etats, compteur, dossier, d_images, t, SL, SO, CL, CA, MOL, T, Thermostat, g, Temp_equilibre, plafond, h)
    
    print('Fin')



##################################################
# Variables globales
##################################################

    # Variables physiques

T_init = 15 # température
h = 0.002 # coefficient de frottement
g = 1 # constante de gravitation
D = 13 # largeur de l'enceinte
E = 13 # hauteur de l'enceinte
F = 13 # profondeur de l'enceinte
Q = 1 # masse du thermostat de Nose-Hover

r_chlore = 0.009
r_site_libre = 0.1
r_site_occupé = 0.1

proba_ad = 0.95
proba_des = 0.002

rayons = [r_site_libre, r_site_occupé, r_chlore]

r = 2.19 # raport nombre chlore/nombre site libre

    # Couples d'etats

L_choc = [(0,0), (2,2), (1,2), (2,1), (0,1), (1,0), (1,1)]
L_adso = [(0,2), (2,0)]
L_rien = [(0,3), (1,3), (2,3), (3,0), (3,1), (3,2), (3,3)]

    # Valeurs des états

# 0 = valeur initiale

# 0 = site libre
# 1 = site occupé
# 2 = molecule de chlore à adsorber
# 3 = molecule de chlore adsorbée

    # Paramètres de la simulation

plafond = True # enceinte fermée en hauteur
Thermostat = True # présence d'un thermostat
N = 200 # nombre de molécules
tmax = 600 # temps de la simulation
tol_deplacement = 0.5 # pas de temps de la simulation
N_images = 3000 # nombre d'images de la simulation
images = True # affichage des graphiques



##################################################
# Lancement de la simulation
##################################################

main(N,rayons,g,T_init,h,D,E,F,plafond,Thermostat,Q,tmax,tol_deplacement,images,N_images)



##################################################
# Fin du programme
##################################################
