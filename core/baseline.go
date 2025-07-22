package core

type BaseLine struct {
	userBias   []float64 //b_u
	itemBias   []float64 // b_i
	globalBias float64
	trainSet   TrainSet
}

func NewBaseLine() *BaseLine {
	return new(BaseLine)
}

func (b *BaseLine) Predict(userID, itemID int) float64 {
	innerUserID := b.trainSet.ConvertUserID(userID)
	innerItemID := b.trainSet.ConvertItemID(itemID)
	ret := b.globalBias
	if innerUserID != newID {
		ret += b.userBias[innerUserID]
	}
	if innerItemID != newID {
		ret += b.itemBias[innerItemID]
	}
	return ret
}
func (b *BaseLine) Fit(trainSet TrainSet, params Parameters) {
	// Setup options
	reader := newParameterReader(params)
	reg := reader.getFloat64("reg", 0.02)
	lr := reader.getFloat64("lr", 0.005)
	nEpochs := reader.getInt("nEpochs", 20)
	b.trainSet = trainSet
	b.userBias = make([]float64, b.trainSet.userCount)
	b.itemBias = make([]float64, b.trainSet.itemCount)

	// baseline算法有两个参数需要训练得到, 使用随机梯度下降算法
	for epoch := 0; epoch < nEpochs; epoch++ {
		for i := 0; i < trainSet.Length(); i++ {
			userID, itemID, rating := trainSet.Users[i], trainSet.Items[i], trainSet.Ratings[i]

			innerUserID := trainSet.ConvertUserID(userID)
			innerItemID := trainSet.ConvertItemID(itemID)
			userBias := b.userBias[innerUserID]
			itemBias := b.itemBias[innerItemID]
			// 计算梯度
			diff := b.Predict(userID, itemID) - rating

			gradGlobalBias := diff
			gradUserBias := diff + reg*userBias
			gradItemBias := diff + reg*itemBias

			// 更新梯度
			b.globalBias -= lr * gradGlobalBias
			b.userBias[innerUserID] -= lr * gradUserBias
			b.itemBias[innerItemID] -= lr * gradItemBias
		}
	}
}
