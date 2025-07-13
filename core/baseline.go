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
	if innerUserID != noBody {
		ret += b.userBias[innerUserID]
	}
	if innerItemID != noBody {
		ret += b.itemBias[innerItemID]
	}
	return ret
}
func (b *BaseLine) Fit(trainSet TrainSet, options ...OptionSetter) {

	// 设置选项
	option := Option{
		reg:     0.02,
		lr:      0.005,
		nEpochs: 20,
	}
	for _, editor := range options {
		editor(&option)
	}
	b.trainSet = trainSet
	b.userBias = make([]float64, b.trainSet.userCount)
	b.itemBias = make([]float64, b.trainSet.itemCount)

	users, items, ratings := trainSet.Interactions()
	// baseline算法有两个参数需要训练得到, 使用随机梯度下降算法
	for epoch := 0; epoch < option.nEpochs; epoch++ {
		for i := 0; i < trainSet.Length(); i++ {
			userID, itemID, rating := users[i], items[i], ratings[i]

			innerUserID := trainSet.ConvertUserID(userID)
			innerItemID := trainSet.ConvertItemID(itemID)
			userBias := b.userBias[innerUserID]
			itemBias := b.itemBias[innerItemID]
			// 计算梯度
			diff := b.Predict(userID, itemID) - rating

			gradGlobalBias := diff
			gradUserBias := diff + option.reg*userBias
			gradItemBias := diff + option.reg*itemBias

			// 更新梯度
			b.globalBias -= option.lr * gradGlobalBias
			b.userBias[innerUserID] -= option.lr * gradUserBias
			b.itemBias[innerItemID] -= option.lr * gradItemBias
		}
	}
}
