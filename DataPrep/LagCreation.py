import platform
import pandas as pd
class lags:
    def __init__(self):
        if platform.system() == "Darwin":
            self.data = pd.read_csv("data/createdDat/TimeBuckets.csv")
        elif platform.system() == "Windows":
            self.data = pd.read_csv("Data\\createdDat\\TimeBuckets.csv")

    def buildLaggedFeatures(self, s, columns, lag=5,dropna=True):
        '''
        From http://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
        Builds a new DataFrame to facilitate regressing over all possible lagged features
        '''

        if type(s) is pd.DataFrame:
            new_dict={}
            for c in s.columns:
                new_dict[c]=s[c]
            for col_name in columns:
                new_dict[col_name]=s[col_name]
                # create lagged Series
                for l in range(1,lag+1):
                    new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
            res=pd.DataFrame(new_dict,index=s.index)

        elif type(s) is pd.Series:
            the_range=range(lag+1)
            res=pd.concat([s.shift(i) for i in the_range],axis=1)
            res.columns=['lag_%d' %i for i in the_range]
        else:
            print('Only works for DataFrame or Series')
            return None
        if dropna:
            return res.dropna()
        else:
            return res 



if __name__ == '__main__':
    l = lags()
    data = l.data

    lagsData = l.buildLaggedFeatures(data, ["Energy (kWh)"])
    print(lagsData.head(10))
