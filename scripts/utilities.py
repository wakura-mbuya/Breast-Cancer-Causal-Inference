import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer, MinMaxScaler

class Utilities:
    """
    This class contains utility functions to help in plotting various plots and perform other useful frequent tasks
    """
    
    def __init__(self):
        """
        Constructor for the Utilities class
        """
        pass
    
    def describe(self, df):
        """
        This function Generates basic descriptive statistical information like mean, median, quartiles on a dataframe

        Args: 
            df: (pandas DataFrame) a dataframe you need to obtain its descriptive statistics

        Returns:
            description: (pandas DataFrame) a dataframe that holds statistical information about the variables
        """
        try:
            description = df.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                                .background_gradient(subset=['std'], cmap='Reds')\
                                .background_gradient(subset=['50%'], cmap='coolwarm')
            return description

        except:
            print("could not generate description")
            
            
            
            
    def normalize(self, df):
        """
        Normalizes a dataframe by making the mean of each variable 0 and their SD 1

        Args:
            df: a dataframe that holds only numerical variables

        Returns:
            normal: a normalized dataframe.
        """
        normald = Normalizer()
        normal = pd.DataFrame(normald.fit_transform(df))
        return normal
    
    def scale(self, df):
        """
        scale variables using min-max scaler to bring all values between 0 and 1
        for each of the variables.

        Args:
            df: a dataframe that holds only numerical variables

        Returns:
            scaled: a dataframe with scaled variables.
        """
        scaler = MinMaxScaler()
        scaled = pd.DataFrame(scaler.fit_transform(df))
        return scaled

    def scale_and_normalize(self, df):
        """
        Runs the scaler and normalizer together and returns scaled and normalized 
        dataframe

        Args: 
            df: a dataframe with only numerical variables

        Returns: 
            normScaled: a dataframe with scaled and normalized variables 
        """
        try:
            columns = df.columns.to_list()
            normScaled = self.normalize(self.scale(df))
            normScaled.columns = columns
            return normScaled
        except:
            print("could not scale and normalize")
            
            ######################################################################################
##                               plotting methods                                   ##
######################################################################################

    def plot_graph(self, sm, th, save=False, name=None):
        """
        plots a structure model or causal graph by not including edges below the th.

        Args:
            sm: a causalnex structure model
            th: a treshold to use as a reference to eleminate some week edges.
            title: title for the image

        Returns: Image object that holds the causal graph

        """
        try:
            path = f"../data/images/{name}"
            tmp = self.apply_treshold(sm, th)
            viz = plot_structure(
                tmp,
                graph_attributes={"scale": "2.5", 'size': 2},
                all_node_attributes=NODE_STYLE.WEAK,
                all_edge_attributes=EDGE_STYLE.WEAK)
            img = Image(viz.draw(format='png'))
            return img

        except:
            print("graph failed to be generated")

    def show_importance(self, model, cols, size, save = False, name = None):
        importance = model.coef_[0]

        f = plt.figure()
        f.set_figwidth(size[0])
        f.set_figheight(size[1])
        plt.bar(x=cols, height=importance)
        if(save):
            path = f"../data/images/{name}"
            plt.savefig(path)
        plt.show()


    def corr(self, x, y, **kwargs):
        """
        calculates a correlation between two variables

        Args:
            x: a list of values
            y: a list of values

        Returns: nothing
        """
        # Calculate the value
        coef = np.corrcoef(x, y)[0][1]
        # Make the label
        label = r'$\rho$ = ' + str(round(coef, 2))
        
        # Add the label to the plot
        ax = plt.gca()
        ax.annotate(label, xy = (0.2, 0.95), size = 11, xycoords = ax.transAxes)

        
    def plot_pair(self, df, title, range, size, save=False, name=None):
        """
        generates a pair plot that shows distribution of one variable and 
        its relationship with other variables using scatter plot.

        Args:
            range: the range of variables to include in the chart
            size: the size of the chart

        Returns: None.
        """
        try:
            target = df["diagnosis"]
            data = df.iloc[:,1:]
            data = pd.concat([target,data.iloc[:,range[0]:range[1]]],axis=1)
            plt.figure(figsize=(size[0],size[1]))
            grid=sns.pairplot(data=data,kind ="scatter",hue="diagnosis",palette="Set1")
            grid.fig.suptitle(title)
            grid = grid.map_upper(self.corr)

            if(save):
                path = f"../data/images/{name}"
                plt.savefig(path)

            plt.show()
        except:
            print("pair-plot failed to be generated")  


    
    def show_corr(self, df, title, size=[17,10], range=None, save=False, name=None):
        """
        plots a correlation matrix heatmap

        Args:
            df: dataframe that holds the data
            size: size of the chart to be plotted
            range: the range of columns or variables to include in the chart

        Returns: None
        """
        try:
            # correlation matrix
            if range is None:
                corr_matrix = df.corr()
            else:
                if(range[1] == -10):
                    corr_matrix = df.iloc[:,range[0]:].corr()
                else:
                    corr_matrix = df.iloc[:,range[0]:range[1]].corr()
            matrix = np.triu(corr_matrix)
            fig, ax = plt.subplots(figsize=(size[0], size[1]))
            plt.title(title)
            ax = sns.heatmap(corr_matrix, annot=True, mask=matrix)

            if(save):
                path = f"../data/images/{name}"
                plt.savefig(path)
        except:
            print("correlation heatmap could not be generated")    

        

    def plot_violin(self, df, size, title, save=False, name=None):
        """
        plots a violin graph

        Args:
            df: a dataframe that holds both the feature and target variables
            size: a list that holds the size of the chart to be plotted
            save: whether to savethe data or not.
            name: name of the chart to save.

        Returns: None
        """
        try:
            df = df.copy()
            df.iloc[:,1:] = self.scale_and_normalize(df.iloc[:,1:]) 
            data = pd.concat([df.iloc[:,:]],axis=1)
            data = pd.melt(data,id_vars="diagnosis",
                                var_name="features",
                                value_name='value')
            plt.figure(figsize=(size[0],size[1]))
            plt.title(title)
            sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart",palette ="Set2")
            plt.xticks(rotation=90)

            if(save):
                path = f"../data/images/{name}"
                plt.savefig(path)
        except:
            print("violin failed")
    
    def plot_bar(self, df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str, ax):
        """
        Plots a bar graph of the dataframe columns
        """
        plt.figure(figsize=(12, 7))
        sns.barplot(data = df, x=x_col, y=y_col, ax=ax)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        return plt.show()
    
    def plot_pie(self, data, column, title:str):
        """
        Plots a pie chart
        """
        plt.figure(figsize=(12, 7))
        count = data[column].value_counts()
        colors = sns.color_palette('pastel')[0:5]
        plt.pie(count, labels = count.index, colors = colors, autopct='%.0f%%')
        plt.title(title, size=18, fontweight='bold')
        return plt.show()
    
    def plot_hist(self, df:pd.DataFrame, column:str, color:str):
        """
        Plots a histogram
        """
        plt.figure(figsize=(9, 7))
        sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        return plt.show()
