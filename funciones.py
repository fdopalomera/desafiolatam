#!/usr/bin/env python
# coding: utf-8


# Importación de librerías a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans


# Función para recodificar variables
def recod_var(dict_var, var, new_var, df):
    
    """
    Función: Recodifica y cambia el nombre la variable categórica ingresada, 
    según un diccionario que contiene los valores de reemplazo y a reemplazar.
    
    Parámetros:
    - dict_var (dict): Diccionario que contiene como'keys' los valores de reemplazo, 
    y como'values' los valores a reemplazar.
    - var (str): Nombre de la ariable a recodificar del DataFrame
    - new_var (str): Nuevo nombre de la variable a recodificar
    - df (DataFrame): DataFrame donde proviene la variable a recodificar
    
    Retorno: Devuelve el número total de observaciones organizadas en las nuevas categorías 
    """
    
    # Creación de nueva variable recodificada
    df[new_var] = df[var]
    for i in dict_var:
        df[new_var].replace(dict_var[i], i, inplace=True)
        
    # Eliminación variable recodificada    
    df.drop(var, axis=1, errors='ignore', inplace=True)
    
    return df[new_var].value_counts()




# Función para Histogramas con Boxplot

def plot_hist(df,var):
    
    """
    Función que genera un histograma con boxplot de la variable var identificando 
    media y mediana.
   
   Parámetros:
    - df (DataFrame): DataFrame con datos.
    - var (str): Variable del dataframe a graficar.
    
    Retorno: Devuelve un Histográma con boxplot de la variable.
    """
    
    #Eliminación de valores perdidos.
    df_tmp = df[var].dropna()
    
    # definición de subplots.
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                        gridspec_kw= {"height_ratios": (0.2, 1)})
    mean = round(df_tmp.mean(), 2)
    median = round(df_tmp.median(), 2)

    #Gráfico de boxplot de referencia.
    sns.boxplot(df_tmp, ax=ax_box)
    ax_box.axvline(mean, color='coral', linestyle='--')
    ax_box.axvline(median, color='crimson', linestyle='-')

    #Gráfico de histograma.
    sns.distplot(df_tmp, ax=ax_hist)
    ax_hist.axvline(mean, color='coral', linestyle='--')
    ax_hist.axvline(median, color='crimson', linestyle='-')

    plt.legend({'Mean: '+str(mean):mean,'Median: '+str(median):median})

    ax_box.set(xlabel=var)
    
    plt.show()
    
    return




# Crear Gráficos según tipo de variable (Dummy, Categórica, Continua)

def univariate_plots(df, categ_max=12, fig_size=(10, 10), plots_per_row=3):
    
    """
    Función: Genera gráficos de distribución de los datos de las variables de 
    un DataFrame, según el tipo de variable.
    
    Parámetros:
    - df (DataFrame): DataFrame a utilizar. 
    - categ_max (int): Número máximo de categorías a considerarse al discriminar
    entre variable categórica y continua. 
    
    Retorno: Gráficos de seaborn y matplotlib.
    """
    
    fig = plt.figure(figsize=fig_size)
    rows = np.ceil(len(df.columns) / plots_per_row)
    
    
    for i, var in enumerate(df):
        
        ax = fig.add_subplot(rows, plots_per_row, i+1)
        n_values = len(df[var].value_counts())
       
        # Graficar histograma para variables continuas
        if n_values > categ_max and not df[var].dtype == np.object:
            sns.distplot(df[var])
        
        elif n_values == 2:
            # Graficar barplot para variables dummy
            ax = sns.countplot(x=var,data=df)
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height + 3,
                        '{:1.2f}'.format(height/len(df[var])), ha="center") 
            
            
        elif n_values <= categ_max:
            # Graficar barplot para variables categóricas
            ax = sns.barplot(x=df[var].value_counts().index, 
                             y=df[var].value_counts().values)
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height + 3,
                        '{:1.2f}'.format(height/len(df[var])), ha="center")
            
            if df[var].dtypes == np.object and len(df[var].dropna().unique()
                ) > 3 and max([len(i) for i in df[var].dropna().unique()]
                ) > 1.5 * len(df[var].dropna().unique()):

                plt.xticks(rotation=75)
    
    plt.tight_layout()
    
    return


# In[1]:


def binarize(df, categ_max=5):
    
    """
    Función: Modifica el DataFrame ingresado al binarizar las variables categóricas, 
    discriminando de variables continuas según número máximo de categorías a 
    considerar ingresado.
    
    Parámetros:
    - df (DataFrame): DataFrame a utilizar. 
    - categ_max (int): Número máximo de categorías a considerarse al discriminar.
    entre variable categórica y continua. 
    
    Retorno: DataFrame procesado resultante del proceso.
    """
    
    df = df.copy()
    
    for var in df:
        n_values = len(df[var].value_counts())
        value_counts = df[var].value_counts().sort_values().index
        
        if n_values == 2: 
            # Creación de nueva variable binarizada en el DataFrame
            df[str(var)+'_'+str(value_counts[0])] = np.where(
                df[var] == value_counts[0] ,1, 0)
            # Remueve variable binarizada
            df.drop(columns=var, inplace=True)
        
        elif n_values <= categ_max:
        
            for i, j in enumerate(value_counts[:-1]):
                # Creación de nueva variable binarizada en el DataFrame
                df[str(var)+'_'+str(value_counts[i])] = np.where(
                    df[var] == j, 1, 0)
            # Remueve variable categórica binarizada
            df.drop(columns=var, inplace=True)
    
    return df


# In[3]:


def heatmap(df, v_max=.3, v_min=-.3, fig_size=(15,15)):

    """
    Función: Genera un gráfico de mapa de calor según las correlaciones lineales
    de las varibles de un DataFrame ingresado.
    
    Parámetros:
    - df (DataFrame): DataFrame a utilizar. 
    - v_max (float): Cota superior de la escala de colores del Heatmap.
    - v_min (float): Cota inferior de la escala de colores del Heatmap
    - fig_size (Tuple): Tupla con la altura y ancho del gráfico. 
    
    Retorno: Gráfico de mapa de calor.
    """

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=fig_size)
    with sns.axes_style("white"):
        ax = sns.heatmap(corr, mask=mask, vmax=v_max, vmin=v_min, square=True, annot=True)
    
    return


# In[2]:


def bivariate_plots(df, target_var, categ_max=5):
    
    """
    Función: Genera gráficos bivariados de una variable objetivo continua en función  
    de las demás variables de un DataFrame, según el tipo de variable.
    
    Parámetros:
    - df (DataFrame): DataFrame a utilizar. 
    - target_var (Series): Vector que contiene variable objetivo.
    - categ_max (int): Número máximo de categorías a considerarse al discriminar
    entre variable categórica y continua. 
    
    Retorno: Serie de Gráficos de seaborn y matplotlib.
    """
    
    df_tmp = df.drop(columns=target_var)
    
    for var in df_tmp:
        n_values = len(df[var].value_counts())
        
        if n_values > categ_max:

            sns.lmplot(var, target_var, df)
            plt.title('Scatterplot entre '+str(target_var)+' y "'+str(var)+'"')

        else:

            plt.figure()
            sns.boxplot(x=var, y=target_var, data=df)
            plt.title('Boxplot de '+str(target_var)+' agrupado por "'+str(var)+'"')
            

    return


# In[ ]:


def bivariate_categ_plots(df, target_var, categ_max=6):
    
    """
    Función: Genera gráficos bivariados de una variable objetivo categórica en función  
    de las demás variables de un DataFrame, según el tipo de variable.
    
    Parámetros:
    - df (DataFrame): DataFrame a utilizar. 
    - target_var (Series): Vector que contiene variable objetivo.
    - categ_max (int): Número máximo de categorías a considerarse al discriminar
    entre variable categórica y continua. 
    
    Retorno: Serie de Gráficos de seaborn y matplotlib.
    """

    for var in df.drop(columns=target_var):

        n_values = len(df[var].value_counts())

        if n_values <= categ_max:
        
            xtab = pd.crosstab(index=df[var], columns=df[target_var])
            xtab = xtab.sort_values(by=0, ascending=False)
            xtab = xtab.applymap(lambda x: np.round(x/len(df), 2)).plot(kind='bar')
            plt.title('Barplot de '+str(target_var)+' según "'+str(var)+'"')
            plt.xticks(rotation=75)

        else:

            plt.figure()
            plt.title('Boxplot entre '+str(target_var)+' y "'+str(var)+'"')
            sns.boxplot(x=target_var, y=var, data=df)
    return


# In[ ]:


def ols_feature_rm(X_mat, y_vec, add_constant=True, p_value_max=.05):
    
    """
    Función: A partir de una matriz de atributos y un vector objetivo, itera
    una modelación de regresión lineal, eliminandose una variable por iteración
    según el criterio de máximo p value de la serie de coeficientes estimados, 
    hasta que no hayan atributos que sea mayor al valor preestablecido (Por defecto, 5%).
    
    Parámetros:
    - X_mat (DataFrame): Matriz de atributos inicial.
    - y_vect (Series): Vector objetivo.
    - add_constant (Boolean): si se desea que la regresión posea un intercepto.
    Valor por defecto = True
    - max_p_value: p value máximo que pueden poseer los coeficientes de la regresión.
    Vaor por defecto = .05

    
    Retorno: Matriz de atributos final (DataFrame)
    """
    
    if add_constant == True:
        X_mat = sm.add_constant(X_mat)
    
    for i in X_mat.columns.tolist():
    
        sm_OLS = sm.OLS(y_vec,X_mat).fit()
        p_value_series = sm_OLS.summary2().tables[1]['P>|t|']
        max_p_value = max(p_value_series)
        
        if max_p_value > p_value_max:
            var2rm = p_value_series[p_value_series.values == max_p_value].index.values
            X_mat = X_mat.drop(columns = var2rm)
                
        else:
            break
            
    tmp_X_Mat = X_mat.drop(columns='const', errors='ignore')
    
    return tmp_X_Mat


# In[ ]:


# Creación de función 'Elbow Graph'

def elbow_graph(X_mat, max_clusters=10, random_state=12345, axvline=5, max_iter=300, n_jobs=-1):

    # generamos un array para guardar los resultados.
    inertia = []

    # Para cada número entre 1 y 10
    for i in range(1, max_clusters+1):
        # Agregamos la inercia 
        inertia.append(KMeans(n_clusters=i, random_state=random_state, 
            max_iter=max_iter, n_jobs=n_jobs).fit(X_mat).inertia_)

    # graficamos el resultado
    plt.plot(range(1, max_clusters+1), inertia, 'o-', color='tomato') 
    plt.xlabel("Cantidad de clusters") 
    plt.ylabel("Inercia")
    plt.title("Elbow graph")
    plt.axvline(axvline)


# Función para filtar el data frame según un valor de una columna específica
def filter_df(df, var, value):
    
    is_value = df[var] == value
    df_tmp = df[is_value]
    
    return df_tmp