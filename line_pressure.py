import pandas as pd
import numpy as np
import pyodbc
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, normalize
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def normal_pull():
	# Trusted connection allows for Windows authentication
	try:
		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
									r'Server=10.75.6.160, 1433;'
									r'Database=OperationsDataMart;'
									r'trusted_connection=yes'
									)
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		DROP TABLE IF EXISTS #Normal

		SELECT	P.Wellkey
				,W.API
				,P.DateKey
				,P.LinePressure
				,P.TubingPressure
				,P.CasingPressure
				,P.MeasuredOil
				,P.MeasuredGas
				,P.MeasuredWater
				,P.LastChokeEntry
				,P.LastChokeStatus
				,ROW_NUMBER() OVER (PARTITION BY P.Wellkey ORDER BY P.DateKey DESC) AS RowNum
		  INTO #Normal
		  FROM [OperationsDataMart].[Facts].[Production] P
		  JOIN [OperationsDataMart].[Dimensions].[Wells] W
			ON W.Wellkey = P.Wellkey
		  WHERE P.LinePressure IS NOT NULL
			AND W.BusinessUnit = 'North'
			AND P.LastChokeStatus = 'Normal Operations'
	""")

	cursor.execute(SQLCommand)

	SQLCommand = ("""
		SELECT  API
				,AVG(LinePressure) AS AvgLine
				,STDEV(LinePressure) AS Deviation
				,AVG(TubingPressure) AS AvgTubing
				,AVG(CasingPressure) AS AvgCasing
		  FROM #Normal
		  WHERE RowNum <= 30
		  GROUP BY API
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
		df.drop_duplicates(inplace=True)
	except:
		df = None
		print('Dataframe is empty')

	return df

def data_pull():
	# Trusted connection allows for Windows authentication
	try:
		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
									r'Server=10.75.6.160, 1433;'
									r'Database=OperationsDataMart;'
									r'trusted_connection=yes'
									)
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		SELECT	P.Wellkey
				,DP.WellFlac
				,DP.API10 AS API
				,DP.DateKey
				,P.LinePressure
				,DP.AllocatedOil
				,DP.AllocatedGas
				,DP.AllocatedWater
				,P.LastChokeEntry
				,P.LastChokeStatus
		  FROM [Business].[Operations].[DailyProduction] DP
		  JOIN [OperationsDataMart].[Dimensions].[Wells] W
			ON W.WellFlac = DP.WellFlac
		  JOIN [OperationsDataMart].[Facts].[Production] P
		    ON P.Wellkey = W.Wellkey
			AND P.DateKey = DP.DateKey
		  WHERE P.LinePressure IS NOT NULL
			AND W.BusinessUnit = 'North'
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
		df.drop_duplicates(inplace=True)
	except:
		df = None
		print('Dataframe is empty')

	return df

def anomaly(df):
	df.sort_values('DateKey', inplace=True)
	df.loc[:, 'DateKey'] = pd.to_datetime(df.loc[:, 'DateKey'])

	df.loc[:, 'rolling'] = df.loc[:, 'LinePressure'].rolling(4, center=False).mean()
	df.loc[:, 'rolling_std'] = df.loc[:, 'LinePressure'].rolling(4, center=False).std()
	df.loc[:, 'rolling_gas'] = df.loc[:, 'AllocatedGas'].rolling(4, center=False).mean()

	X = df['rolling_std'].values
	X_1 = df['rolling'].values
	X_gas = df['rolling_gas'].values

	y_pred = np.ones(X.shape)
	y_pred[X > 35] = 0

	pressure_vals = df.loc[:, 'LinePressure'].values.reshape(-1, 1)
	X_pred = linear(df.loc[:, 'DateKey'].astype('int64').values.reshape(-1, 1),
					pressure_vals)
	# std = np.std(df.loc[:, 'LinePressure'].values)
	# upper = (X_pred + (1.96 * std)).reshape(-1, 1)
	# lower = (X_pred - (1.96 * std)).reshape(-1, 1)

	std = np.std(df.loc[df['LinePressure'] > 0, 'LinePressure'].values)
	med = df.loc[df['LinePressure'] > 0, 'LinePressure'].median()
	upper = med + std * .5
	lower = med - std * .5

	df['spike'] = np.where(df['rolling'] > upper, 1, 0)
	df['3_spike'] = np.where((((df['spike'] == df['spike'].shift(1)) &
							   (df['spike'] == df['spike'].shift(2))) |
							  ((df['spike'] == df['spike'].shift(-1)) &
						  	   (df['spike'] == df['spike'].shift(-2))) |
							  ((df['spike'] == df['spike'].shift(1)) &
						  	   (df['spike'] == df['spike'].shift(-1)))) &
							  df['spike'] == 1, 1, 0)

	df['5_spike'] = np.where((((df['spike'] == df['spike'].shift(-4)) &
							   (df['spike'] == df['spike'].shift(-3)) &
							   (df['spike'] == df['spike'].shift(-2)) &
							   (df['spike'] == df['spike'].shift(-1))) |
							  ((df['spike'] == df['spike'].shift(-3)) &
  							   (df['spike'] == df['spike'].shift(-2)) &
  							   (df['spike'] == df['spike'].shift(-1)) &
  							   (df['spike'] == df['spike'].shift(1))) |
							  ((df['spike'] == df['spike'].shift(-2)) &
  							   (df['spike'] == df['spike'].shift(-1)) &
  							   (df['spike'] == df['spike'].shift(1)) &
  							   (df['spike'] == df['spike'].shift(2))) |
							  ((df['spike'] == df['spike'].shift(-1)) &
  							   (df['spike'] == df['spike'].shift(1)) &
  							   (df['spike'] == df['spike'].shift(2)) &
  							   (df['spike'] == df['spike'].shift(3))) |
							  ((df['spike'] == df['spike'].shift(1)) &
  							   (df['spike'] == df['spike'].shift(2)) &
  							   (df['spike'] == df['spike'].shift(3)) &
  							   (df['spike'] == df['spike'].shift(4)))) &
							  df['spike'] == 1, 1, 0)

	df['spike_start'] = np.where((df['3_spike'] == 1) &
								 ((df['3_spike'].shift(1) == 0) |
								  (df['3_spike'].shift(1) == np.nan)),
								 1, 0)

	out_pressure = (pressure_vals > upper).astype(int) + (pressure_vals < lower).astype(int)

	df.loc[:, 'press_norm'] = normalize(df['LinePressure'].values.reshape(1, -1))[0]
	df.loc[:, 'gas_norm'] = normalize(df['AllocatedGas'].values.reshape(1, -1))[0]

	spike_df = pd.DataFrame(columns=['api', 'start_date', 'length', 'pre_prod',
									 'deferment', 'avg_spike_prod', 'daily_def',
									 'norm_def'])
	for idx, row in enumerate(df.iterrows()):
		if idx != 0:
			target_gas = df.iloc[idx-1]['AllocatedGas']
			target_norm = df.iloc[idx-1]['gas_norm']
			if row[1]['spike_start'] == 1:
				spike_gas = []
				spike_norm = []
				for index, spike_row in enumerate(df.iloc[idx:].iterrows()):
					if spike_row[1]['AllocatedGas'] >= target_gas:
						break
					else:
						spike_gas.append(spike_row[1]['AllocatedGas'])
						spike_norm.append(spike_row[1]['gas_norm'])
				deferment = sum(np.array(spike_gas) - target_gas)
				def_norm = sum(np.array(spike_norm) - target_norm)

				spike_df = spike_df.append(pd.DataFrame(
										   np.array([row[1]['API'],
										   			 row[1]['DateKey'], len(spike_gas),
												     target_gas, abs(deferment),
													 np.mean(spike_gas),
													 np.mean(abs(deferment)),
													 np.mean(abs(def_norm))]).reshape(1, -1),
										   columns=['api', 'start_date', 'length',
													'pre_prod', 'deferment',
													'avg_spike_prod', 'daily_def',
													'norm_def']))

	# plot_linear(df, X_pred, upper, lower, df['API'].unique()[0])
	# plot_it(df, X, X_1, X_gas, y_pred, df['API'].unique()[0])

	# rf_regressor(df.loc[df['5_spike'] == 1, :])

	return df, spike_df

# June 11th - 22nd
# Normalize each well, then run regression on the whole set

def rf_regressor(df):
	X = df['press_norm'].values.reshape(-1, 1)
	# X_norm = normalize(X)
	y = df['gas_norm'].values.reshape(-1, 1)
	# y_norm = normalize(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y,
														test_size=0.2,
														random_state=13)

	rf = RandomForestRegressor()

	rf.fit(X_train, y_train)
	print('Score')
	print(rf.score(X_test, y_test))

def linear(X, y):
	scaler = StandardScaler().fit(X, y)

	X_scale = scaler.transform(X)

	poly = PolynomialFeatures(degree=2)
	X_ = poly.fit_transform(X)

	# model = make_pipeline(PolynomialFeatures(3), LinearRegression())
	# model.fit(X, y)

	lr = LinearRegression()
	lr.fit(X_, y)

	return lr.predict(X_).flatten()

def plot_linear(df, line, upper, lower, api):
	plt.close()
	fig, ax = plt.subplots()

	std = np.std(df.loc[df['LinePressure'] > 0, 'LinePressure'].values)
	med = df.loc[df['LinePressure'] > 0, 'LinePressure'].median()
	upper = med + std * .5
	lower = med - std * .5

	ax.plot(df['DateKey'], df['rolling'], color='black', linestyle='--', alpha=.3)
	ax.plot(df.loc[(df['rolling'] <= upper) & (df['rolling'] >= lower), 'DateKey'],
			df.loc[(df['rolling'] <= upper) & (df['rolling'] >= lower),'rolling'],
			color='black', label='Line Pressure')
	ax.plot(df['DateKey'], df['rolling_gas'], color='#42d9f4')
	# linestyle='--', alpha=.3)
	# ax.plot(df.loc[(df['LinePressure'] <= upper) & (df['LinePressure'] >= lower), 'DateKey'],
	# 		df.loc[(df['LinePressure'] <= upper) & (df['LinePressure'] >= lower),'rolling_gas'],
	# 		color='#42d9f4', label='Gas')

	# df.loc[:, 'flag_up'] = (df.loc[:, 'LinePressure'] > upper).astype(int)
	# df.loc[:, 'flag_down'] = (df.loc[:, 'LinePressure'] < lower).astype(int)
	# df.loc[:, 'flag'] = df.loc[:, ['flag_up', 'flag_down']].sum(axis=1)
	#
	# for i in range(0, len(df['DateKey'].values)):
	# 	ax.axvspan(i, i+1, color='red', alpha=df.loc[:, 'flag'].values[i]/4)

	ax.axhline(upper, color='b', linestyle='--')
	ax.axhline(lower, color='b', linestyle='--')
	ax.axhline(med, color='r')

	plt.legend()
	plt.title('{} Linear STD'.format(api))

	plt.savefig('figures/{}_linear.png'.format(api))

def plot_it(df, X, X_1, X_gas, y_pred, api):
	plt.close()

	fig, ax = plt.subplots()
	ax.plot(X_1, label='Line Pressure')
	ax.plot(X_gas, label='Gas', color='black')
	for i in range(0, len(X)):
		ax.axvspan(i, i+1, color='red', alpha=((y_pred[i]-1)*-1)/4)
	# ax.plot(flagged * 100, 'xb')
	plt.ylabel('Line Pressure (PSI)')
	plt.xlabel('Day')
	plt.legend()
	plt.title('{} 14 day rolling'.format(api))
	plt.savefig('figures/{}_14rolling.png'.format(api))

def plot_corr(df):
	plt.close()

	fig,  ax = plt.subplots()
	ax.scatter(df['press_norm'], df['gas_norm'])

	plt.ylabel('Normalized Gas')
	plt.xlabel('Normalized Line Pressure')
	plt.title('Correlation of Gas and Pressure During Spikes')

	plt.savefig('figures/correlation.png')

def neural_net(df):
	df.sort_values('DateKey', inplace=True)
	df.loc[:, 'line_sq'] = df.loc[:, 'LinePressure'] ** 2
	df.loc[:, 'line_cu'] = df.loc[:, 'LinePressure'] ** 3
	df.loc[:, 'rolling'] = df.loc[:, 'LinePressure'].rolling(4, center=False).mean()
	df.loc[:, 'rolling_std'] = df.loc[:, 'LinePressure'].rolling(4, center=False).std()

	df = delta(df)
	df.fillna(0, inplace=True)
	# df.loc[:, 'line_change'] = df.loc[:, 'LinePressure'] - df.loc[:, 'LinePressure'].shift(1)
	# df.loc[:, 'line_change'].fillna(0, inplace=True)

	y = df['AllocatedGas']
	X = df[['LinePressure', 'rolling', 'rolling_std', 'line_sq', 'line_cu',
			'line_change']]

	mlp = MLPRegressor()
	mlp.fit(X, y)

	print('Score: {}'.format(mlp.score(X, y)))

def delta(df):
	return_df = pd.DataFrame(columns=df.columns)

	for api in df['API'].unique():
		well_df = df.loc[df['API'] == api]
		well_df.loc[:, 'line_change'] = well_df.loc[:, 'LinePressure'] - \
										well_df.loc[:, 'LinePressure'].shift(1)
		well_df.loc[:, 'line_change'].fillna(0, inplace=True)
		return_df = return_df.append(well_df)

	return return_df

def build_spike(flacs, df):
	kelvin_df = df.loc[df['WellFlac'].isin(flacs), :]
	non_kelvin_df = df.loc[~df['WellFlac'].isin(flacs), :]

	apis = sorted(kelvin_df['API'].unique())
	full_df = pd.DataFrame()
	spike_df = pd.DataFrame()
	nonk_spike_df = pd.DataFrame()

	for api in apis:
		api_df, api_spike = anomaly(kelvin_df.loc[kelvin_df['API'] == api,
									  ['API', 'DateKey', 'LinePressure', 'AllocatedGas']])
		full_df = spike_df.append(api_df)
		spike_df = spike_df.append(api_spike)

	for api in sorted(non_kelvin_df['API'].unique()):
		api_df, api_spike = anomaly(non_kelvin_df.loc[non_kelvin_df['API'] == api,
									  ['API', 'DateKey', 'LinePressure', 'AllocatedGas']])
		full_df = spike_df.append(api_df)
		nonk_spike_df = nonk_spike_df.append(api_spike)

	spike_df.loc[:, 'length'] = spike_df.loc[:, 'length'].astype(float)
	spike_df.loc[:, 'deferment'] = spike_df.loc[:, 'deferment'].astype(float)
	spike_df.loc[:, 'daily_def'] = spike_df.loc[:, 'daily_def'].astype(float)
	spike_df.loc[:, 'norm_def'] = spike_df.loc[:, 'norm_def'].astype(float)

	nonk_spike_df.loc[:, 'length'] = nonk_spike_df.loc[:, 'length'].astype(float)
	nonk_spike_df.loc[:, 'deferment'] = nonk_spike_df.loc[:, 'deferment'].astype(float)
	nonk_spike_df.loc[:, 'daily_def'] = nonk_spike_df.loc[:, 'daily_def'].astype(float)
	nonk_spike_df.loc[:, 'norm_def'] = nonk_spike_df.loc[:, 'norm_def'].astype(float)

	return spike_df, nonk_spike_df


def plot_spike(kelv_df, non_df):
	plt.close()

	fig, ax = plt.subplots(1, 1, figsize=(5, 5))

	max_len = np.max([non_df['length'].max(), kelv_df['length'].max()])

	ax.scatter(non_df['length'], non_df['norm_def'], s=5,
			   color='#429bf4', label='Pre Kelvin')
	non_fit = np.polyfit(non_df['length'].values, non_df['norm_def'].values, 1)
	non_fit_fn = np.poly1d(non_fit)
	ax.plot(np.append(non_df['length'], max_len), non_fit_fn(np.append(non_df['length'], max_len)),
			linestyle='--', color='#429bf4')
	print('Non Kelvin R2 of {}'.format(r2_score(non_df['norm_def'], non_fit_fn(non_df['length']))))

	ax.scatter(kelv_df['length'], kelv_df['norm_def'], s=5,
			   color='#f48342', label='Post Kelvin')
	kelv_fit = np.polyfit(kelv_df['length'].values, kelv_df['norm_def'].values, 1)
	kelv_fit_fn = np.poly1d(kelv_fit)
	ax.plot(np.append(kelv_df['length'], max_len), kelv_fit_fn(np.append(kelv_df['length'], max_len)),
			linestyle='--', color='#f48342')
	print('Kelvin R2 of {}'.format(r2_score(kelv_df['norm_def'], kelv_fit_fn(kelv_df['length']))))

	ax.set_xlabel('Length of Spike (Days)')
	ax.set_ylabel('Deferment per Day (mcfd)')
	plt.title('Comparing Deferment During Line Pressure Spikes')
	plt.legend()
	plt.tight_layout()

	plt.savefig('figures/kelv_comp_normalized.png')


if __name__ == '__main__':
	# df = data_pull()
	# df.to_csv('data/line_pressure.csv')
	df = pd.read_csv('data/line_pressure.csv')
	df = df.loc[(df['LinePressure'].notnull()) & (df['AllocatedGas'].notnull()), :]

	flacs = pd.read_csv('data/kelvin_wellflacs.csv', header=None).values.flatten()
	# kelvin_df, non_kelvin_df = build_spike(flacs, df)

	kelvin_df.to_csv('data/kelvin_spike.csv')
	non_kelvin_df.to_csv('data/non_kelvin_spike.csv')
	kelvin_df = pd.read_csv('data/kelvin_spike.csv')
	non_kelvin_df = pd.read_csv('data/non_kelvin_spike.csv')

	# neural_net(df)

	# plot_corr(spike_df)
	# rf_regressor(spike_df.loc[spike_df['3_spike'] == 1, :])

	plot_spike(kelvin_df.loc[(kelvin_df['avg_spike_prod'].notnull()) &
							 (kelvin_df['start_date'] >= pd.Timestamp(2018, 2, 1)), :],
			   kelvin_df.loc[(kelvin_df['avg_spike_prod'].notnull()) &
			   				 (kelvin_df['start_date'] < pd.Timestamp(2018, 2, 1)) &
							 (kelvin_df['length'] < 100), :])

	plot_spike(kelvin_df.loc[(kelvin_df['avg_spike_prod'].notnull()) &
							 (kelvin_df['start_date'] >= pd.Timestamp(2018, 2, 1)), :],
			   non_kelvin_df.loc[non_kelvin_df['avg_spike_prod'].notnull(), :])
