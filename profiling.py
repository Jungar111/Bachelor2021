from pandas_profiling import ProfileReport
from Palo_Alto_Initial import clean_paloalto

c = clean_paloalto()
data = c.clean_data()

profile = ProfileReport(data, title='Pandas Profiling Report', explorative=True)
profile.to_file("Pandas_Profiling_Report.html")