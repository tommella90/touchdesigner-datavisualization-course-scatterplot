import os 
import pandas as pd
import numpy as np
import TDFunctions as tdf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


def normalize(col, range):
    newcol = ((col - col.min() ) / (col.max() - col.min() ) * range) 
    return newcol


def TableToList(col):
	new_ls = []
	for i in range(1, len(col)):
		new_ls.append(float(col[i].val))
	new_ls = np.array(new_ls)
	return new_ls


def TableToPandas(self_col):
	col = op('blueprint/df_all').col(self_col)
	new_ls = []

	for i in range(1, len(col)):
		new_ls.append(col[i].val)
	series = pd.Series(new_ls)

	try:
		series = series.astype(float)
	except:
		temp_list = series.unique()
		len_list = list(range(len(temp_list)))
		series.replace(temp_list, len_list, inplace=True)

	return series


def TableToDataFrame(table):
	table = op('blueprint/k_vars')
	kvars = []
	n = table.numRows
	df_k = []

	for i in range(0, table.numRows-1):
		name = str(table[i,0].val)
		col = pd.DataFrame(op('blueprint/df_all').col(name))
		col = col.iloc[1:,]
		col = normalize(col,1)
		df_k.append(col)
	df_k = pd.concat(df_k, axis=1)

	return df_k



class scatterPlot:

	def __init__(self, myOp):
		self.myOp = myOp
		self.debug = myOp.par.Debug
	## saver parameters
		self.reset = myOp.par.Resetdefault
		self.title = myOp.par.Title
		self.folder = myOp.par.Folder
		self.save = myOp.par.Save
		self.frm = myOp.par.Fileformat
		self.replace = myOp.par.Replace
	## plot parameters
		self.load = myOp.par.Loaddata
		self.df = myOp.par.Dataframe 
		self.marker_size = myOp.par.Markersize
		self.x = myOp.par.X
		self.y = myOp.par.Y
		self.color_group = myOp.par.Colorgroup
		self.size_group = myOp.par.Sizegroup
		self.color_type = myOp.par.Colortype
		self.custom_colors = myOp.par.Choosecustomcolors
		self.name = myOp.par.Name
	## OUTLIERS
		self.out_hx = myOp.par.Outlierhighx
		self.out_lx = myOp.par.Outlierlowx 
		self.out_hy = myOp.par.Outlierhighy
		self.out_ly = myOp.par.Outlierlowy
	## ESTIMATIONS
		self.fit_ols = myOp.par.Fitols
		self.num_k = myOp.par.Numk
		self.kmean_vars = myOp.par.Kmeanvars
		self.ols_group = myOp.par.Olsgroup
		self.show_group_ols = myOp.par.Showgroupols
		self.ols_outlierscolorsr = myOp.par.Olsoutlierscolorr
	## Picker parameters
		self.details1 = myOp.par.Details1
		self.see_details = myOp.par.Seedetails
	## thicks
		self.thicks = myOp.par.Thicks
		self.thicks_num = myOp.par.Thicksnum

	## Set default parameters
		#attr = op('attrAssign')
		#attr.par.name0 = 'size'
		#attr.par.value0 = 0.007
	## OLS 

	## debug
		print("working")
		return


	def DefaultAttributes(self): 
		BASE = op('blueprint')
		if self.reset == True:
			BASE.par.X, BASE.par.Y = "", ""
			BASE.par.Markersize = 0.007
			BASE.par.Markercolorr, BASE.par.Markercolorg, BASE.par.Markercolorb = .05, .16, .65 
			BASE.par.Colorgroup = "None"
			BASE.par.Olscolorr, BASE.par.Olscolorg, BASE.par.Olscolorb = .5, 0, 0
			BASE.par.Showols = 0
			BASE.par.Dropxhigh, BASE.par.Dropxlow, BASE.par.Dropyhigh, BASE.par.Dropylow = 100, 100, 100, 100
			BASE.par.Title, BASE.par.Titlesize = "", 35
			BASE.par.Titlebackgroundr, BASE.par.Titlebackgroundg, BASE.par.Titlebackgroundb, BASE.par.Titlebackgrounda = .9, .9, .9, 0
			BASE.par.Labelsize = 18
			BASE.par.Thicks, BASE.par.Thicksnum, BASE.par.Thickssize = 1, 5, 17
			BASE.par.Backgroundcolorr, BASE.par.Backgroundcolorg, BASE.par.Backgroundcolorb, BASE.par.Backgroundcolora = .9, .9, .9, 1
			BASE.par.Colortype, BASE.par.Applycustomcolors = 0, 0 
			BASE.par.Sizegroup = "None"
			BASE.par.Showdetails, BASE.par.Details1 = 0, ""
			BASE.par.Legend = 0
			BASE.par.Name, BASE.par.Folder, BASE.par.Fileformat, BASE.par.Replace = "", "", "png", 1
			BASE.par.Numk, BASE.par.Olsgroup = 3, ""
			BASE.par.Outlierhighx, BASE.par.Outlierlowx, BASE.par.Outlierhighy, BASE.par.Outlierlowx = 100, 0, 100, 0
			BASE.par.Olsoutlierscolorr, BASE.par.Olsoutlierscolorg, BASE.par.Olsoutlierscolorb, BASE.par.Olsoutlierscolora = .6, .6, .6, 0
			BASE.par.Showgroupols = 0
			BASE.par.Showolsoutliers = 0
			BASE.par.Folder, BASE.par.Name, BASE.par.Fileformat = "Myfolder", "Mytitle", "png"


	def DefineVariables(self):
		data = pd.read_csv(f"{self.df}")

		num_vars = data.select_dtypes(include='number').columns
		op('blueprint/kmean_vars/numeric_vars').unstore('*')
		op('blueprint/kmean_vars/numeric_vars').store('num_vars', num_vars)

		none = np.ones((len(data.index), 1), dtype=int)
		op('blueprint/none').clear()
		op('blueprint/none').appendRows(none)

		op('blueprint/index').clear()
		op('blueprint/index').appendCol(data.index,0)

		data.insert(0, "None", none) 
		menu_list = list(data.columns)
		menu_list2 = menu_list.copy()
		menu_list_k = menu_list.copy()
		menu_list_k.insert(1, "kmeans") 
		menu_list.pop(0)

		op('blueprint/vars').clear()
		op('blueprint/vars').appendRows(menu_list)

		my_menu = tdf.parMenu(menu_list, menuLabels=None)

		self.x.menuNames = menu_list 
		self.y.menuNames = menu_list

		self.x.menuLabels = menu_list 
		self.y.menuLabels = menu_list

		self.color_group.menuNames = menu_list2
		self.size_group.menuNames = menu_list2

		self.color_group.menuLabels = menu_list2
		self.size_group.menuLabels = menu_list2

		self.details1.menuNames = menu_list_k
		self.details1.menuLabels = menu_list_k

		self.ols_group.menuNames = menu_list_k
		self.ols_group.menuLabels = menu_list_k

		my_vars = [self.x, self.y, self.color_group, self.size_group, self.details1, self.ols_group]

	## Store colors
		n_color_groups = data[f'{self.color_group}'].unique()
		if len(n_color_groups) <= 25:
			pass
		else:
			pass


	def SaveImage(self):
		path = os.getcwd()

		## define file format
		image_saver = op('blueprint/image_saver')
		if f"{self.frm}" != "jpg": 
			image_saver.par.imagefiletype = f"{self.frm}"
		elif f"{self.frm}" == "jpg":
			image_saver.par.imagefiletype = "jpeg"

		file = str(f"{self.folder}" + "/" + f"{self.name}" + f".{self.frm}")
		file = file.replace("/ ", "\ ")		
		file = file.replace(" ", "")

		## save file
		myDir = f"{self.folder}"
		myDir = myDir.replace("/ ", "\ ")
		op('directory').par.rootfolder = myDir


		mylist = op('directory').col("name")
		saver = op('blueprint/button_saver')
		#image_saver.par.file = file

		if self.save == True:
			if f"{self.name}" in str(mylist) and self.replace == False:
				print("FILE ALREADY EXISTS")
				op('blueprint/image_saver').click()
			else:
				saver.click()
				print("saved")


	def CustomColors(self):
		#c = TableToPandas(f"{self.color_group}")

		#n_color_groups = c.unique()
		#op('blueprint/n_color_group').unstore('*')

		if self.color_type == "cont" and self.custom_colors == True:
			op('blueprint/color_cont').openViewer()
			op('blueprint/legend/cat_cont').par.index = 1
			op('blueprint/color_type').par.index = 1

		elif self.color_type == "cat" and self.custom_colors == True:
			op('blueprint/color_categories').openViewer()
			op('blueprint/legend/cat_cont').par.index = 0
			op('blueprint/color_type').par.index = 2

		elif self.color_type == "kmeans" and self.custom_colors == True:
			op('blueprint/color_kmeans').openViewer()
			op('blueprint/legend/cat_cont').par.index = 2
			op('blueprint/color_type').par.index = 3

		pass


	def DrawThicks(self):
		if self.thicks == False:
			op('blueprint/thicks_valuesX').allowCooking = False
			op('blueprint/thicks_valuesY').allowCooking = False
			op('blueprint/thickX').allowCooking = False
			op('blueprint/thickY').allowCooking = False


		elif self.thicks == True:
			op('blueprint/thicks_valuesX').allowCooking = True
			op('blueprint/thicks_valuesX/replicator').par.recreateall.pulse()
			op('blueprint/thicks_valuesY').allowCooking = True
			op('blueprint/thicks_valuesY/replicator').par.recreateall.pulse()			
			op('blueprint/thickX').allowCooking = True
			op('blueprint/thickY').allowCooking = True
			op('blueprint/thicks_valuesX/button1').click()
			op('blueprint/thicks_valuesY/button1').click()

			pass
		pass


	def ResetThicks(self):
		op('blueprint/thicks_valuesY/button1').click()
		op('blueprint/thicks_valuesX/button1').click()		
		print('v')
	

	def FitOLS(self):
		line = op('blueprint/fit_line')

		x = normalize(TableToPandas(f"{self.x}"), 1).to_numpy().reshape(-1,1)
		y = normalize(TableToPandas(f"{self.y}"), 1)

		regressor = LinearRegression()  
		model = regressor.fit(x,y)
		beta, alpha = model.coef_, model.intercept_

		line.par.pax, line.par.pay = 0, alpha
		line.par.pbx, line.par.pby = 1, beta + alpha


	def GroupOlsColor(self):
		g = TableToPandas(f"{self.ols_group}")
		ols_cat = list(g.unique())
		ols_cat = [int(s) for s in ols_cat]	

		print(ols_cat)
		#op('blueprint/ols_group_color').clear()
		#op('blueprint/ols_group_color').appendRows(ols_cat)
		

	def FitGroupOls(self):
		x = normalize(TableToPandas(f"{self.x}"), 1)
		print(x)
		y = normalize(TableToPandas(f"{self.y}"), 1)
		g = TableToPandas(f"{self.ols_group}")

		if self.ols_group == "kmeans":
			op('blueprint').par.Colortype = 3

		df = pd.concat([x, y, g], axis=1)
		levels = g.unique()
		levels = sorted(levels)
		regressor = LinearRegression()  
		op('blueprint/beta').clear()
		op('blueprint/alpha').clear()

		for i in levels:
			local_df = df[df[2] == i]
			ind = local_df[0].to_numpy().reshape(-1,1)
			dep = local_df[1].to_numpy()
			model = regressor.fit(ind, dep)
			beta, alpha = model.coef_, [model.intercept_]
			op('blueprint/beta').appendRows(beta)
			op('blueprint/alpha').appendRows(alpha)

		op('blueprint/button1').click()

		## choose color group (kmeans or choosen group)
		if op('blueprint').par.Olsgroup == "kmeans":
			op('blueprint/switch4').par.index == 0
		else:
			op('blueprint/switch4').par.index == 0
		

	def ManageOutliers(self):
		line = op('blueprint/fit_outliers')
		x = normalize(TableToPandas(f"{self.x}"), 1)
		y = normalize(TableToPandas(f"{self.y}"), 1)

		tot, one = x.size, x.size/100
		x_h, x_l, y_h, y_l = self.out_hx, self.out_lx, self.out_hy, self.out_ly

		high_x, low_x = np.percentile(x, x_h), np.percentile(x, x_l)
		op('blueprint/high_x').par.pax, op('blueprint/high_x').par.pbx = high_x, high_x
		op('blueprint/low_x').par.pax, op('blueprint/low_x').par.pbx = low_x, low_x

		high_y, low_y = np.percentile(y, y_h), np.percentile(y, y_l)
		op('blueprint/high_y').par.pay, op('blueprint/high_y').par.pby = high_y, high_y
		op('blueprint/low_y').par.pay, op('blueprint/low_y').par.pby = low_y, low_y

		df = pd.DataFrame({'x': x, 'y': y})
		df = df.loc[(df['x']>low_x) & (df['x']<high_x) & (df['y']>low_y) & (df['y']<high_y)]

		x = df['x'].to_numpy().reshape(-1,1)
		y = df['y'].to_numpy()

		regressor = LinearRegression()  
		model = regressor.fit(x,y)
		beta, alpha = model.coef_, model.intercept_
		print(model.coef_, model.intercept_)

		line.par.pax, line.par.pay = 0, alpha
		line.par.pbx, line.par.pby = 1, beta + alpha

		area = op('blueprint/df_limit')
		area.par.sizex = high_x - low_x
		area.par.sizey = high_y - low_y
		area.par.tx = (high_x + low_x)/2
		area.par.ty = (high_y + low_y)/2


	def ChooseKmeanVars(self):
		op('blueprint/kmean_vars').openViewer()


	def FitKmeans(self):
		k = TableToDataFrame(op('blueprint/num_vars'))
		k = normalize(k, 1)
		
		n = int(self.num_k)
		num_k = list(range(0, n))
		op('blueprint/n_k').clear()		
		op('blueprint/n_k').appendRow(num_k)

		kmeans = KMeans(n_clusters=n, random_state=101010)
		kmeans.fit(k)
		clusters = kmeans.predict(k).tolist()

		op('blueprint/kmeans').clear()
		op('blueprint/kmeans').appendRows(clusters)


	def SeeDetails(self):
		op('blueprint/win_details').openViewer()
		op('blueprint').par.Legend = False
		print('deta')


	def UpdateColors(self):
		## To manage TD bug: geo colors don't update automatically for some reas
		op('blueprint/button1').click()
		pass	

	def Stocazzo(self):
		x = TableToDataFrame(op('blueprint/df_ready'))
		print(x)










		











'''

	def ApplyDetails(self):
		data = op('blueprint/final_data')
		if data.findCell(f"{self.details1}") == f"{self.details1}": 
			print('yes')

		else: 
			print('no')
			pass


	def PrepareData(self):
		row_data = pd.read_csv(f"{self.df}")
		variables = row_data.columns

		def normalize(col, range):
		    newcol = ((col - col.min() ) / (col.max() - col.min() ) * range) - range/2
		    return newcol

	## open data	   
		row_data = pd.read_csv(f"{self.df}")
		variables = row_data.columns
		op('blueprint/variables_store').unstore('*')
		op('blueprint/variables_store').store('variables', variables)
		df = row_data[[f"{self.x}", f"{self.y}"]]
		
		parameters = [f"{self.color_group}", f"{self.size_group}", f"{self.details1}",]

		for par in parameters: 
			if par in variables:
				df = df.merge(row_data[par], left_index=True, right_index=True)

			elif par not in variables and par != "": 
				print(f"{par}", ":  VAR NAME MISPELLED")

			elif par =="": 
				#print(par, "vuoto")
				pass

		df = df.dropna()
		df[f"{self.x}"] = normalize(df[f"{self.x}"], 2)
		df[f"{self.y}"] = normalize(df[f"{self.y}"], 2)
		#df[f"{self.details1}"] = normalize(df[f"{self.y}"], 2)
		df[f"{self.size_group}"] = normalize(df[f"{self.y}"], 1)

		try:
			df.rename(columns={f"{self.color_group}": 'color_group'}, inplace=True, errors='raise')
			num_colors = df['color_group'].unique().max() + 1
			num_samples = len(df.index)
			op('blueprint/color_group').par.resolutionw = num_colors
			op('blueprint/group_color_assign').par.resolutionw = num_samples
			#print(num_colors, num_samples)

		except:
			#print("No color group")
			pass

		df.to_csv(f'{self.folder}/temp_table.csv')
		op('blueprint/reload').click()
		return df
'''


		#print(str(f"{self.color_group}"), df.columns)
		
'''
		if str(f"{self.color_group}") in df.columns: 
			print('ok')
		else: 
			print('no')
		pass
			color_table = pd.read_csv("C:/Users/tomma/Videos/Material/colors/color_table.csv")
			color_table = color_table[["interior", "r", "g", "b"]]
			color_cat = df[f"{self.color_group}"].unique()
			df[f"{self.color_group}"] = df[f"{self.color_group}"] - 1
			num_colors = len(color_cat)	
			color_table = color_table.iloc[0: len(color_cat), :]
			color_table.reset_index(inplace=True, drop=True)
			df = df.merge(color_table, left_on=f"{self.color_group}", right_index=True)

'''



	##m Normalize values

		#df[f"{self.group_size}"] = Normalize2(df[f"{self.group_size}"], 1) 

	## save in temporary table and load



		#print(type(model.coef_))
		#print(x)

'''
		if displayerY.textWidth < displayerY.par.resolutionw: 
			displayerY.par.fontautosize = 0
			#print(displayerY.textWidth, displayerY.par.resolutionw, displayerY.par.fontautosize)
		elif displayerY.textWidth >= displayerY.par.resolutionw: 
			displayerY.par.fontautosize = 1	
			#print(displayerY.textWidth, displayerY.par.fontautosize)
		pass
'''
'''
		for D in displayers:
			if D.textWidth < D.par.resolutionw: 
				D.par.fontautosize = 0
				print(D.textWidth)
				print(D.par.fontautosize)

			elif D.textWidth >= D.par.resolutionw: 
				D.par.fontautosize = 1
				print(D.textWidth)
				print(D.par.fontautosize)

'''





'''
def onEvents(renderPickDat, events, eventsPrev):

	for event, eventPrev in zip(events, eventsPrev):
		if event.select and event.pickOp: 
			if '/2wayArea/geo_minT/' in event.pickOp.path or '/2wayArea/geo_maxT/' in event.pickOp.path:
				mouse_x = event.texture.x
				op('constant_u').par.value0 = mouse_x
				op('constant_select').par.value0 = 1

				if '/2wayArea/geo_min' in event.pickOp.path: 
					op('constant_geo').par.value0 = 0 
				if '/2wayArea/geo_max' in event.pickOp.path: 
					op('constant_geo').par.value0 = 1 

		if event.selectEnd:		
			op('constant_select').par.value0 = 0 

		pass

	return


	def CreateDataframe(self):
		self.myOp.DefineVariables()
		df, header = op('blueprint/df'), op('blueprint/header')
		df.clear()
		header.clear()
		data = pd.read_csv(f"{self.df}")
		columns = [f"{self.x}", f"{self.y}", f"{self.color_group}", f"{self.size_group}"]
		
		variables = []
		for col in columns:
			if col in data.columns:
				variables.append(col)
			elif col not in data.columns:
				print("not id df")
	

		variables = [data[f"{self.x}"], data[f"{self.y}"], data[f"{self.color_group}"], data[f"{self.size_group}"]]

		for v in range(len(variables)):
			header.appendRow(columns[v])
			op('blueprint/df').appendCol(variables[v])

		## add details
		details1 = data[f"{self.details1}"]
		df.appendCol(details1)
		header.appendRow(f"{self.details1}")

		## index
		df.appendCol(data.index)
		header.appendRow("index")


	## COLOR GROUPS
		n_color_groups = data[f"{self.color_group}"].unique()
		op('blueprint/update_color').click()	
		op('blueprint/n_color_group').unstore('*')
		op('blueprint/n_color_group').store('n_color_groups', n_color_groups)

		pass
'''