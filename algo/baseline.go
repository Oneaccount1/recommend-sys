package algo

import (
	"recommend-sys/data"
)

type Baseline struct {
	userBias   map[int]float64
	itemBias   map[int]float64
	globalBias float64
}

func NewBaseLineRecommender() *Baseline {
	return &Baseline{}
}

func (b *Baseline) Predict(userID, itemID int) float64 {
	userBias, _ := b.userBias[userID]
	itemBias, _ := b.itemBias[itemID]
	return userBias + itemBias + b.globalBias
}
func (b *Baseline) Fit(trainSet data.Set, options ...OptionsEditor) {
	b.userBias = make(map[int]float64)
	b.itemBias = make(map[int]float64)

	for _, userID := range trainSet.AllUsers() {
		b.userBias[userID] = 0
	}

	for _, itemID := range trainSet.AllItems() {
		b.itemBias[itemID] = 0
	}

	// 设置选项
	option := Option{
		regularization: 0.02,
		learningRate:   0.005,
		nEpoch:         20,
	}

	for _, editor := range options {
		editor(&option)
	}

	users, items, ratings := trainSet.AllInteraction()
	// baseline算法有两个参数需要训练得到, 使用随机梯度下降算法
	for epoch := 0; epoch < option.nEpoch; epoch++ {
		for i := 0; i < trainSet.NRow(); i++ {
			userID := users[i]
			itemID := items[i]
			rating := ratings[i]
			userBias := b.userBias[userID]
			itemBias := b.itemBias[itemID]
			// 计算梯度
			diff := rating - b.globalBias - userBias - itemBias

			gradGlobalBias := -2 * diff
			gradUserBias := -2*diff + 2*option.regularization*userBias
			gradItemBias := -2*diff + 2*option.regularization*itemBias

			// 更新梯度
			b.globalBias -= option.learningRate * gradGlobalBias
			b.userBias[userID] -= option.learningRate * gradUserBias
			b.itemBias[itemID] -= option.learningRate * gradItemBias
		}
	}
}
