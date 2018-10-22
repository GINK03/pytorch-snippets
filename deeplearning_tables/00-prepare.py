import pandas as pd
import numpy as np

def main():
	dfTrain = pd.read_csv('train.csv')

	yTrain = dfTrain["totals.transactionRevenue"]
	yTrain = yTrain / yTrain.max()
	#yTrain = ( yTrain / yTrain.max() ).values

	dfTrain = dfTrain.drop(['visitStartTime', 'totals.transactionRevenue'], axis=1)

	for col in dfTrain.columns:
		dfTrain[col] = dfTrain[col].rank()
		dfTrain[col] = dfTrain[col] / dfTrain[col].max()
	XTrain = dfTrain.values
	np.savez('train.npz', yTrain=yTrain, XTrain=XTrain)
if __name__ == '__main__':
	main()		
