# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:18:49 2022

@author: J61182
"""


# ========================================
# External imports
# ========================================
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from math import isnan
# ========================================
# Internal imports
# ========================================

from UI.MplCanvas import MplCanvas
from common_functions import show_figure, round_it



def colormap_sorties_matlab(emplacement,proj,methode,vmin,vmax,quiver_scale):
        # Main parameters
        plt.rcParams.update({'font.size': 14})

        
        ############## Plot Matlab Fig ###########
        ##########################################
        
        r_bathy = np.transpose(pd.read_excel(emplacement + 'bathy.xlsx', header=None).to_numpy())
        r_poly = pd.read_excel(emplacement + 'velocity_' + proj + '_' + methode + '.xlsx', header=None).to_numpy()
        r_arrow = pd.read_excel(emplacement + 'arrow_' + proj + '_' + methode + '.xlsx', header=None).to_numpy()
        
        # La section obtenue avec Matlab est centrée en 0. On décale donc pour que le 0 soit
        # sur une des deux rives (rives gauche généralement)      
        
        # On décale donc le fond
        decalage_bathy = -r_bathy[0,0]
        for i in range(len(r_bathy[:,0])):
            r_bathy[i,0] += decalage_bathy            
        
        # De même on décale les cellules puis les flèches
        decalage_vertices = -r_poly[0,0]
        for i in range(len(r_poly[:,0])):
            for j in range(7):
                r_poly[i,j]+= decalage_vertices
                
        for i in range(len(r_arrow[:,0])):
            r_arrow[i,0] += decalage_vertices
        
        
        poly_vert = []
        patches = []
        list_poly_vertices = []
        vect_vel = -r_poly[:,14]    #sens inversé dans Matlab actuellement, il faut donc ce signe -
        
        for i in range(len(r_poly[:,1])):
            for j in range(7):
                poly_vert.append([r_poly[i,j],-r_poly[i,j+7]])    #profondeur négative dans Matlab mais on l'a veut positive dans Python
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))

        
        # on normalise les vitesses pour pouvoir attribuer une couleur à chaque cellule
        norm_vel = [i - vmin for i in vect_vel]/(vmax - vmin)
        poly_colors = []
        line_width = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))
            if isnan(vel):
                line_width.append(0)
            else :
                line_width.append(1)

        patchColl = PatchCollection(patches,facecolors='white') #,edgecolors='black',linewidths = line_width)
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        

        depth = np.max(-r_bathy[:,1])
        
        middle_poly_X = r_arrow[:,0]
        middle_poly_Y = -r_arrow[:,1]
        
        v = r_arrow[:,2]
        w = r_arrow[:,3]
        
        q = fig.ax.quiver(middle_poly_X, middle_poly_Y, v, w, units='xy',
                          scale=quiver_scale,width=0.015 * depth)  # ,width=quiver_width)
        fig.ax.quiverkey(q, X=1, Y=-0.03, U=np.round(quiver_scale, 2), label=str(quiver_scale) + 'm/s',
                          labelpos='S', coordinates='axes', fontproperties={'size': 16})
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_poly[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_poly[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=vmin, vmax=vmax))
        if proj == 'rozovskii':
            if methode == 'mixte':
                cb.ax.set_ylabel(canvas.tr("Vitesse Primaire Rozovskii / Methode Mixte (m/s)"))                 
                cb.ax.yaxis.label.set_fontsize(16)
                cb.ax.tick_params(labelsize=16)
                        
                fig.ax.set_xlim(left=lower_limit, right=upper_limit)
                canvas.draw()
                        
                show_figure(fig)           
                plt.title("Colormap de la projection de Rozovskii avec la méthode mixte")
            else:
                cb.ax.set_ylabel(canvas.tr("Vitesse Primaire Rozovskii / Methode Vermeulen (m/s)"))            
                cb.ax.yaxis.label.set_fontsize(16)
                cb.ax.tick_params(labelsize=16)
                        
                fig.ax.set_xlim(left=lower_limit, right=upper_limit)
                canvas.draw()
                        
                show_figure(fig)
                plt.title("Colormap de la projection de Rozovskii avec la méthode Vermeulen")
            
        else:
            if methode == 'mixte':
                cb.ax.set_ylabel(canvas.tr("Streamwise velocity / Methode Mixte (m/s)"))            
                cb.ax.yaxis.label.set_fontsize(16)
                cb.ax.tick_params(labelsize=16)
                        
                fig.ax.set_xlim(left=lower_limit, right=upper_limit)
                canvas.draw()
                        
                show_figure(fig)
                plt.title("Méthode Mixte")
            
            else:
                cb.ax.set_ylabel(canvas.tr("Streamwise velocity / Methode Vermeulen (m/s)"))            
                cb.ax.yaxis.label.set_fontsize(16)
                cb.ax.tick_params(labelsize=16)
                        
                fig.ax.set_xlim(left=lower_limit, right=upper_limit)
                canvas.draw()
                        
                show_figure(fig)
                plt.title("Méthode Vermeulen")

        
        
        plt.savefig('Images_ppt/Velocity_' + proj + '_' + methode + '.png',bbox_inches='tight',pad_inches=0.1)
        
        return
        
    


def jet(x):
        """ Transformer un nombre entre 0 et 1 en un vecteur RGB selon l'échelle de couleur "jet"
        Parameters
        ----------
        x : float
        Nombre compris entre 0 et 1 
        """
        if x<0:
            r = 0
            v = 0
            b = 0.5
        elif 0.125>x:
            r = 0
            v = 0
            b = 0.5+4*x
        elif 0.375>x>=0.125:
            r=0
            v=4*(x-0.125)
            b=1
        elif 0.625>x>=0.375:
            r=4*(x-0.375)
            v=1
            b=-4*(x-0.625)
        elif 0.875>x>=0.625:
            r=1
            v=-4*(x-0.875)
            b=0
        elif isnan(x):
            r=1
            v=1
            b=1
        elif 1>x>=0.875:
            r = -4*(x-1.125)
            b=0
            v=0
        else:
            r=0.5
            b=0
            v=0
        return [r,v,b]
        


def colormap_ecarts_matlab(emplacement,proj):

        
        r_bathy = np.transpose(pd.read_excel(emplacement + 'bathy.xlsx', header=None).to_numpy())
        r_poly_verm = pd.read_excel(emplacement + 'velocity_' + proj + '_' + 'verm' + '.xlsx', header=None).to_numpy()
        r_poly_mixte = pd.read_excel(emplacement + 'velocity_' + proj + '_' + 'mixte' + '.xlsx', header=None).to_numpy()
        
        
        
        
        decalage_bathy = -r_bathy[0,0]
        for i in range(len(r_bathy[:,0])):
            r_bathy[i,0] += decalage_bathy
        
        decalage_vertices = -r_poly_verm[0,0]
        for i in range(len(r_poly_verm[:,0])):
            for j in range(7):
                r_poly_verm[i,j]+= decalage_vertices

        
        # p_max = 100
        poly_vert = []
        patches = []
        list_poly_vertices = []     #ATTENTION AU NAN !
        # vect_vel = np.abs((r_poly_verm[:,14] - r_poly_mixte[:,14])/np.nanmean(r_poly_verm[:,14])*100)
        vect_vel = -(r_poly_verm[:,14] - r_poly_mixte[:,14])
        
        for i in range(len(r_poly_verm[:,1])):
            for j in range(7):
                poly_vert.append([r_poly_verm[i,j],-r_poly_verm[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        # norm_vel = np.array([i for i in vect_vel])/p_max
        
        vect_vel_0 = [0 if isnan(x) else x for x in vect_vel]
        
        zscore_vel = np.abs(sc.stats.zscore(vect_vel_0))
        
        vel_cor = []

        for i in range(len(vect_vel)):
            if zscore_vel[i]<2:
                vel_cor.append(vect_vel_0[i])
        
        norm_vel = np.array([i-min(vel_cor) for i in vect_vel])/(max(vel_cor)-min(vel_cor))


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_poly_verm[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_poly_verm[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=min(vel_cor), vmax=max(vel_cor)))
        if proj == 'rozovskii':
            cb.ax.set_ylabel(canvas.tr("Ecart en Primary Velocity (ROZ) (m/s)"))
        else:
            cb.ax.set_ylabel(canvas.tr("Différence en Streamwise Velocity (m/s)"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        
        plt.title("Streamwise velocity : Méthode Vermeuleun - Méthode Mixte (m/s)")
        
        plt.savefig('Images_ppt/Ecart_Vel.png',bbox_inches='tight',pad_inches=0.1)
        
        

def colormap_nombre_info(emplacement,nb_max,methode):

        
        r_bathy = np.transpose(pd.read_excel(emplacement + 'bathy.xlsx', header=None).to_numpy())
        r_info_cell = pd.read_excel(emplacement + 'info_cell_'+ methode +'.xlsx', header=None).to_numpy()
        
        
        
        decalage_bathy = -r_bathy[0,0]
        for i in range(len(r_bathy[:,0])):
            r_bathy[i,0] += decalage_bathy
        
        decalage_vertices = -r_info_cell[0,0]
        for i in range(len(r_info_cell[:,0])):
            for j in range(7):
                r_info_cell[i,j]+= decalage_vertices
        

        
        poly_vert = []
        patches = []
        list_poly_vertices = []   
        vect_vel = r_info_cell[:,14]
        
        for i in range(len(r_info_cell[:,1])):
            for j in range(7):
                poly_vert.append([r_info_cell[i,j],-r_info_cell[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        # norm_vel = np.array([i for i in vect_vel])/p_max
        
        
        norm_vel = np.array([i for i in vect_vel])/(nb_max)


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_info_cell[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_info_cell[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=nb_max))
        cb.ax.set_ylabel(canvas.tr("Nombre d'informations"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        if methode == 'verm':    
            plt.title("Nombre d'informations par cellule pour la méthode Vermeulen")
        elif methode == 'mixte':
            plt.title("Nombre d'informations par cellule pour la méthode Mixte")
        
        plt.savefig('Images_ppt/Info_cell_'+ methode +'.png',bbox_inches='tight',pad_inches=0.1)
        
        
        
def colormap_r2(emplacement):

        
        r_bathy = np.transpose(pd.read_excel(emplacement + 'bathy.xlsx', header=None).to_numpy())
        r_r2_verm = pd.read_excel(emplacement + 'r2verm.xlsx', header=None).to_numpy()
        r_r2_class = pd.read_excel(emplacement + 'r2mixte.xlsx', header=None).to_numpy()
        
        
        decalage_bathy = -r_bathy[0,0]
        for i in range(len(r_bathy[:,0])):
            r_bathy[i,0] += decalage_bathy
        
        decalage_vertices = -r_r2_verm[0,0]
        for i in range(len(r_r2_verm[:,0])):
            for j in range(7):
                r_r2_verm[i,j]+= decalage_vertices
                r_r2_class[i,j]+= decalage_vertices

                

        
        poly_vert = []
        patches = []
        list_poly_vertices = []   
        vect_vel = r_r2_verm[:,14]
        
        for i in range(len(r_r2_verm[:,1])):
            for j in range(7):
                poly_vert.append([r_r2_verm[i,j],-r_r2_verm[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        
        
        vect_vel_0 = [0 if isnan(x) else x for x in vect_vel]
        
        zscore_vel = np.abs(sc.stats.zscore(vect_vel_0))
        
        vel_cor = []

        for i in range(len(vect_vel)):
            if zscore_vel[i]<2:
                vel_cor.append(vect_vel_0[i])
        
        p_max = np.max(vel_cor)  
        
        
        norm_vel = np.array([i for i in vect_vel])/(p_max)


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_r2_verm[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_r2_verm[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=p_max))
        cb.ax.set_ylabel(canvas.tr("Erreur quadratique moyenne"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        
        plt.title("Erreur quadratique moyenne avec la méthode Vermeulen")
        
        plt.savefig('Images_ppt/r2_verm.png',bbox_inches='tight',pad_inches=0.1)
        
        
        
        
        
        ################ Mixte ##############
        
        poly_vert = []
        patches = []
        list_poly_vertices = []   
        vect_vel = r_r2_class[:,14]
        
        for i in range(len(r_r2_class[:,1])):
            for j in range(7):
                poly_vert.append([r_r2_class[i,j],-r_r2_class[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        vect_vel = [0 if isnan(x) else x for x in vect_vel]
        
        
        norm_vel = np.array([i for i in vect_vel])/(p_max)


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_r2_class[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_r2_class[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=p_max))
        cb.ax.set_ylabel(canvas.tr("Erreur quadratique moyenne"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        
        plt.title("Erreur quadratique moyenne avec la méthode Mixte")
        
        plt.savefig('Images_ppt/r2_mixte.png',bbox_inches='tight',pad_inches=0.1)


def colormap_sqrt_r2(emplacement):

        
        r_bathy = np.transpose(pd.read_excel(emplacement + 'bathy.xlsx', header=None).to_numpy())
        r_r2_verm = pd.read_excel(emplacement + 'r2verm.xlsx', header=None).to_numpy()
        r_r2_class = pd.read_excel(emplacement + 'r2mixte.xlsx', header=None).to_numpy()
        
        
        decalage_bathy = -r_bathy[0,0]
        for i in range(len(r_bathy[:,0])):
            r_bathy[i,0] += decalage_bathy
        
        decalage_vertices = -r_r2_verm[0,0]
        for i in range(len(r_r2_verm[:,0])):
            for j in range(7):
                r_r2_verm[i,j]+= decalage_vertices
                r_r2_class[i,j]+= decalage_vertices

                

        
        poly_vert = []
        patches = []
        list_poly_vertices = []   
        vect_vel = np.sqrt(r_r2_verm[:,14])
        
        for i in range(len(r_r2_verm[:,1])):
            for j in range(7):
                poly_vert.append([r_r2_verm[i,j],-r_r2_verm[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        
        
        vect_vel_0 = [0 if isnan(x) else x for x in vect_vel]
        
        zscore_vel = np.abs(sc.stats.zscore(vect_vel_0))
        
        vel_cor = []

        for i in range(len(vect_vel)):
            if zscore_vel[i]<2:
                vel_cor.append(vect_vel_0[i])
        
        p_max = np.max(vel_cor)  
        
        
        norm_vel = np.array([i for i in vect_vel])/(p_max)


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_r2_verm[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_r2_verm[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=p_max))
        cb.ax.set_ylabel(canvas.tr("Racine de l'erreur quadratique moyenne"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        
        plt.title("Racine de l'erreur quadratique moyenne avec la méthode Vermeulen")
        
        plt.savefig('Images_ppt/sqrt_r2_verm.png',bbox_inches='tight',pad_inches=0.1)
        
        
        
        
        
        ################ Mixte ##############
        
        poly_vert = []
        patches = []
        list_poly_vertices = []   
        vect_vel = np.sqrt(r_r2_class[:,14])
        
        for i in range(len(r_r2_class[:,1])):
            for j in range(7):
                poly_vert.append([r_r2_class[i,j],-r_r2_class[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        vect_vel = [0 if isnan(x) else x for x in vect_vel]
        
        
        norm_vel = np.array([i for i in vect_vel])/(p_max)


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_r2_class[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_r2_class[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=p_max))
        cb.ax.set_ylabel(canvas.tr("Racine de l'erreur quadratique moyenne"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        
        plt.title("Racine de l'erreur quadratique moyenne avec la méthode Mixte")
        
        plt.savefig('Images_ppt/sqrt_r2_classique.png',bbox_inches='tight',pad_inches=0.1)


def colormap_sig(emplacement):

        
        r_bathy = np.transpose(pd.read_excel(emplacement + 'bathy.xlsx', header=None).to_numpy())
        r_sig_verm = pd.read_excel(emplacement + 'sigverm.xlsx', header=None).to_numpy()
        r_sig_class = pd.read_excel(emplacement + 'sigmixte.xlsx', header=None).to_numpy()
        
        
        decalage_bathy = -r_bathy[0,0]
        for i in range(len(r_bathy[:,0])):
            r_bathy[i,0] += decalage_bathy
        
        decalage_vertices = -r_sig_verm[0,0]
        for i in range(len(r_sig_verm[:,0])):
            for j in range(7):
                r_sig_verm[i,j]+= decalage_vertices
                r_sig_class[i,j]+= decalage_vertices

                

        
        poly_vert = []
        patches = []
        list_poly_vertices = []   
        vect_vel = r_sig_verm[:,14]
        
        for i in range(len(r_sig_verm[:,1])):
            for j in range(7):
                poly_vert.append([r_sig_verm[i,j],-r_sig_verm[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        
        
        vect_vel_0 = [0 if isnan(x) else x for x in vect_vel]
        
        zscore_vel = np.abs(sc.stats.zscore(vect_vel_0))
        
        vel_cor = []

        for i in range(len(vect_vel)):
            if zscore_vel[i]<2:
                vel_cor.append(vect_vel_0[i])
        
        p_max = np.max(vel_cor)  
        
        
        norm_vel = np.array([i for i in vect_vel])/(p_max)


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_sig_verm[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_sig_verm[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=p_max))
        cb.ax.set_ylabel(canvas.tr("Ecart type sigma"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        
        plt.title("Ecart type avec la méthode Vermeulen")
        
        plt.savefig('Images_ppt/sig_verm.png',bbox_inches='tight',pad_inches=0.1)
        
        
        
        
        
        ################ Mixte ##############
        
        poly_vert = []
        patches = []
        list_poly_vertices = []   
        vect_vel = r_sig_class[:,14]
        
        for i in range(len(r_sig_class[:,1])):
            for j in range(7):
                poly_vert.append([r_sig_class[i,j],-r_sig_class[i,j+7]])
            list_poly_vertices.append(poly_vert)
            poly_vert=[]
        
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        vect_vel = [0 if isnan(x) else x for x in vect_vel]
        
        
        norm_vel = np.array([i for i in vect_vel])/(p_max)


        poly_colors = []
        
        for vel in norm_vel:
            poly_colors.append(jet(vel))

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.plot(r_bathy[:,0], -r_bathy[:,1], color='k', linewidth=2)
        fig.ax.plot(r_bathy[:,0], [0 for i in r_bathy[:,1]], color='b', linewidth=2)
        
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(-r_sig_class[:,8]) * 1.1)
        lower_limit = - 0.5
        upper_limit = r_sig_class[-1,3] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=p_max))
        cb.ax.set_ylabel(canvas.tr("Ecart type sigma"))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
        
        show_figure(fig)
        
        
        plt.title("Ecart type avec la méthode Mixte")
        
        plt.savefig('Images_ppt/sig_classique.png',bbox_inches='tight',pad_inches=0.1)

        
def plot_dispersion_vitesse(emplacement):
            
        r_poly_verm = pd.read_excel(emplacement + 'velocity_streamwise_' + 'verm' + '.xlsx', header=None).to_numpy()
        r_poly_mixte = pd.read_excel(emplacement + 'velocity_streamwise_' + 'mixte' + '.xlsx', header=None).to_numpy()
        r_arrow_verm = pd.read_excel(emplacement + 'arrow_streamwise_' + 'verm' + '.xlsx', header=None).to_numpy()
        r_arrow_mixte = pd.read_excel(emplacement + 'arrow_streamwise_' + 'mixte' + '.xlsx', header=None).to_numpy()
            
        v_x_verm = r_poly_verm[:,-1]
        v_x_clas = r_poly_mixte[:,-1]
        v_y_verm = r_arrow_verm[:,2]
        v_y_clas = r_arrow_mixte[:,2]
        v_z_verm = r_arrow_verm[:,-1]
        v_z_clas = r_arrow_mixte[:,-1]
        
        v_x_verm = [0 if abs(x)>15 else x for x in v_x_verm]        
        v_x_clas = [0 if abs(x)>15 else x for x in v_x_clas]
        v_y_verm = [0 if abs(x)>15 else x for x in v_y_verm]        
        v_y_clas = [0 if abs(x)>15 else x for x in v_y_clas]
        v_z_verm = [0 if abs(x)>15 else x for x in v_z_verm]        
        v_z_clas = [0 if abs(x)>15 else x for x in v_z_clas]
        
        
        v_x_verm_0 = [0 if isnan(x) else x for x in v_x_verm]
        
        v_x_clas_0 = [0 if isnan(x) else x for x in v_x_clas]

        x_vel_verm = np.abs(sc.stats.zscore(v_x_verm_0))
        x_vel_clas = np.abs(sc.stats.zscore(v_x_clas_0))

        for i in range(len(v_x_verm)):
            if x_vel_verm[i]>1 or x_vel_clas[i]>1:
                v_x_verm[i] = 0
                v_x_clas[i] = 0
                
                
        v_y_verm_0 = [0 if isnan(x) else x for x in v_y_verm]
        
        v_y_clas_0 = [0 if isnan(x) else x for x in v_y_clas]

        y_vel_verm = np.abs(sc.stats.zscore(v_y_verm_0))
        y_vel_clas = np.abs(sc.stats.zscore(v_y_clas_0))

        for i in range(len(v_y_verm)):
            if y_vel_verm[i]>1 or y_vel_clas[i]>1:
                v_y_verm[i] = 0
                v_y_clas[i] = 0
                
        
        v_z_verm_0 = [0 if isnan(x) else x for x in v_z_verm]
        
        v_z_clas_0 = [0 if isnan(x) else x for x in v_z_clas]

        z_vel_verm = np.abs(sc.stats.zscore(v_z_verm_0))
        z_vel_clas = np.abs(sc.stats.zscore(v_z_clas_0))

        for i in range(len(v_z_verm)):
            if z_vel_verm[i]>1 or z_vel_clas[i]>1:
                v_z_verm[i] = 0
                v_z_clas[i] = 0

        
        min_x = min(np.nanmin(v_x_verm),np.nanmin(v_x_clas))
        max_x = max(np.nanmax(v_x_verm),np.nanmax(v_x_clas))
        min_y = min(np.nanmin(v_y_verm),np.nanmin(v_y_clas))
        max_y = max(np.nanmax(v_y_verm),np.nanmax(v_y_clas))
        min_z = min(np.nanmin(v_z_verm),np.nanmin(v_z_clas))
        max_z = max(np.nanmax(v_z_verm),np.nanmax(v_z_clas))

        plt.figure()
        plt.scatter(v_x_verm,v_x_clas,s=10)
        plt.plot([min_x,max_x],[min_x,max_x],'black')
        plt.title('Dispersion de la vitesse normale V_x')
        plt.xlabel('V_x méthode Vermeulen (m/s)')
        plt.ylabel('V_x méthode mixte (m/s)')
        plt.savefig('Images_ppt/Dispersion_vel_x.png',bbox_inches='tight',pad_inches=0.1)
        plt.show()
            
        plt.figure()
        plt.scatter(v_y_verm,v_y_clas,s=10,c='coral')
        plt.plot([min_y,max_y],[min_y,max_y],'black')
        plt.title('Dispersion de la vitesse transverse V_y')
        plt.xlabel('V_y méthode Vermeulen (m/s)')
        plt.ylabel('V_y méthode mixte (m/s)')
        plt.savefig('Images_ppt/Dispersion_vel_y.png',bbox_inches='tight',pad_inches=0.1)
        plt.show()
        
        plt.figure()
        plt.scatter(v_z_verm,v_z_clas,s=10,c='b')
        plt.plot([min_z,max_z],[min_z,max_z],'black')
        plt.title('Dispersion de la vitesse verticale V_z')
        plt.xlabel('V_z méthode Vermeulen (m/s)')
        plt.ylabel('V_z méthode mixte (m/s)')
        plt.savefig('Images_ppt/Dispersion_vel_z.png',bbox_inches='tight',pad_inches=0.1)
        plt.show()
            
            