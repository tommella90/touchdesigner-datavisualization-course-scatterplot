# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# 
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.
GENERATOR = op('blueprint')


def onValueChange(par, prev):
	outliers = ['Outlierhighx', 'Outlierlowx', 'Outlierhighy', 'Outlierlowy']
	variables = ['Choosecolorgroup', 'Choosesizegroup', 'details1', 'Colorgroup', 'Sizegroup']

	GENERATOR.FitOLS()

	if par.name == "Dataframe":
		GENERATOR.DefineVariables()
	elif par.name == "Thicksnum":
		GENERATOR.DrawThicks()
	elif par.name in outliers:
		GENERATOR.ManageOutliers()
	elif par.name in variables:
		GENERATOR.DefineVariables()
	elif par.name == "Colortype":
		GENERATOR.CustomColors()
	elif par.name == "Numk":
		GENERATOR.FitKmeans()
		if op('blueprint').par.Olsgroup == "kmeans":
			GENERATOR.FitGroupOls()
	elif par.name == "Olsgroup":
		GENERATOR.GroupOlsColor()
		GENERATOR.FitGroupOls()
		GENERATOR.UpdateColors()
	elif par.name == "Olsoutlierscolorr" or par.name == "Olsoutlierscolorg" or par.name == "Olsoutlierscolorb" or par.name == "Olsoutlierscolora":
		GENERATOR.UpdateColors()
	elif par.name == "Thicksnum":
		GENERATOR.ResetThicks()	
	elif par.name == "Seedetails":
		GENERATOR.SeeDetails()	

	return



# Called at end of frame with complete list of individual parameter changes.
# The changes are a list of named tuples, where each tuple is (Par, previous value)
def onValuesChanged(changes):
	for c in changes:
		#print(par.name)
	# use par.eval() to get current value
		par = c.par
		prev = c.prev
		if c.par.name == "Thicks":
			GENERATOR.DrawThicks()

		if c.par.name == "Customcolors":
			if c.prev == True:
				op('blueprint/switch_cont').par.index = 0
				op('blueprint/color_categories/switch_cat').par.index = 0
				GENERATOR.CustomColors()

			if c.prev == False:
				op('blueprint/switch_cont').par.index = 1
				op('blueprint/color_categories/switch_cat').par.index = 1
				GENERATOR.CustomColors()

	return


def onPulse(par):
	
	if par.name == "Resetdefault":
		GENERATOR.DefaultAttributes()
	elif par.name == "Debug":
		GENERATOR.Stocazzo()
	elif par.name == "Save":
		GENERATOR.SaveImage()
	elif par.name == "Fitols":
		GENERATOR.FitOLS()
		GENERATOR.UpdateColors()		
	elif par.name == "Loaddata":
		GENERATOR.DefineVariables()
	elif par.name == "Manageoutliers":
		GENERATOR.ManageOutliers()
	elif par.name == "Fitkmeans":
		GENERATOR.FitKmeans()
	elif par.name == "Choosecustomcolors":
		GENERATOR.CustomColors()
	elif par.name == "Kmeanvars":
		GENERATOR.ChooseKmeanVars()
	elif par.name == "Updatecolors":
		GENERATOR.UpdateColors()
	elif par.name == "Seedetails":
		GENERATOR.SeeDetails()	
	elif par.name == "Fitolsbygroup":
		GENERATOR.GroupOlsColor()
		GENERATOR.FitGroupOls()
		GENERATOR.UpdateColors()

	return

def onExpressionChange(par, val, prev):
	return

def onExportChange(par, val, prev):
	return

def onEnableChange(par, val, prev):
	return

def onModeChange(par, val, prev):
	return
	