package core

type Baseline struct {
	userBias   map[int]float64
	itemBias   map[int]float64
	globalBias float64
}

type stu struct {
	Name string
}

func NewBaseLine() *Baseline {

	return new(Baseline)

}

func (b *Baseline) Predict(userID, itemID int) float64 {
	userBias, _ := b.userBias[userID]
	itemBias, _ := b.itemBias[itemID]
	return userBias + itemBias + b.globalBias
}
func (b *Baseline) Fit(trainSet TrainSet, options ...OptionSetter) {

	// 设置选项
	option := Option{
		reg:     0.02,
		lr:      0.005,
		nEpochs: 20,
	}

	b.userBias = make(map[int]float64)
	b.itemBias = make(map[int]float64)

	for userID := range trainSet.Users() {
		b.userBias[userID] = 0
	}

	for itemID := range trainSet.Items() {
		b.itemBias[itemID] = 0
	}
	for _, editor := range options {
		editor(&option)
	}

	users, items, ratings := trainSet.Interactions()
	// baseline算法有两个参数需要训练得到, 使用随机梯度下降算法
	for epoch := 0; epoch < option.nEpochs; epoch++ {
		for i := 0; i < trainSet.Length(); i++ {
			userID := users[i]
			itemID := items[i]
			rating := ratings[i]
			userBias := b.userBias[userID]
			itemBias := b.itemBias[itemID]
			// 计算梯度
			diff := rating - b.globalBias - userBias - itemBias

			gradGlobalBias := -2 * diff
			gradUserBias := -2*diff + 2*option.reg*userBias
			gradItemBias := -2*diff + 2*option.reg*itemBias

			// 更新梯度
			b.globalBias -= option.lr * gradGlobalBias
			b.userBias[userID] -= option.lr * gradUserBias
			b.itemBias[itemID] -= option.lr * gradItemBias
		}
	}
}
