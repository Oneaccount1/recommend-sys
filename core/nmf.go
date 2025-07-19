//  基于NMF的协同过滤算法， 与SVD非常类似

package core

import "gonum.org/v1/gonum/floats"

type NMF struct {
	userFactor [][]float64 // p_u
	itemFactor [][]float64 // q_i
	trainSet   TrainSet    // 训练数据集
}

func (N *NMF) Predict(userId int, itemId int) float64 {
	innerUserID := N.trainSet.ConvertUserID(userId)
	innerItemID := N.trainSet.ConvertItemID(itemId)
	if innerUserID != newID && innerItemID != newID {
		return floats.Dot(N.userFactor[innerUserID], N.itemFactor[innerItemID])
	}
	return 0
}

func (N *NMF) Fit(trainSet TrainSet, options Options) {
	nFactors := options.GetInt("nFactors", 15)
	nEpochs := options.GetInt("nEpochs", 50)
	initLow := options.GetFloat64("initLow", 0)
	initHigh := options.GetFloat64("initHigh", 1)
	reg := options.GetFloat64("reg", 0.06)
	//lr := options.GetFloat64("lr", 0.005)
	N.trainSet = trainSet
	// 初始化参数
	N.userFactor = newUniformMatrix(trainSet.UserCount(), nFactors, initLow, initHigh)
	N.itemFactor = newUniformMatrix(trainSet.ItemCount(), nFactors, initLow, initHigh)

	// 创建Buffer
	buffer := make([]float64, nFactors)
	// 创建用于存储乘法更新规则分子和分母的中间矩阵
	userUp := newZeroMatrix(trainSet.UserCount(), nFactors)
	userDown := newZeroMatrix(trainSet.UserCount(), nFactors)
	itemUp := newZeroMatrix(trainSet.ItemCount(), nFactors)
	itemDown := newZeroMatrix(trainSet.ItemCount(), nFactors)

	// 获取数据
	users, items, ratings := trainSet.Interactions()

	for epoch := 0; epoch < nEpochs; epoch++ {
		// 重置中间矩阵
		resetZeroMatrix(userUp)
		resetZeroMatrix(userDown)
		resetZeroMatrix(itemUp)
		resetZeroMatrix(itemDown)

		// 计算中间矩阵
		for i := 0; i < len(ratings); i++ {
			userID, itemID, rating := users[i], items[i], ratings[i]
			innerUserID := trainSet.ConvertUserID(userID)
			innerItemID := trainSet.ConvertItemID(itemID)
			prediction := N.Predict(userID, itemID)

			// 更新userUp (用户因子更新公式的分子部分: Σ(r_ui * q_i))
			copy(buffer, N.itemFactor[innerItemID])
			// buffer = rating * q_i
			mulConst(rating, buffer)
			// userUp[u] += rating * q_i
			floats.Add(userUp[innerUserID], buffer)

			// 更新userDown (用户因子更新公式的分母部分: Σ(p_u^T * q_i * q_i) + λ * p_u)
			copy(buffer, N.itemFactor[innerItemID])
			// buffer = (p_u^T * q_i) * q_i
			mulConst(prediction, buffer)
			// userDown[u] += prediction * q_i
			floats.Add(userDown[innerUserID], buffer)

			copy(buffer, N.userFactor[innerUserID])
			// buffer = reg * p_u (正则化项)
			mulConst(reg, buffer)
			// userDown[u] += reg * p_u
			floats.Add(userDown[innerUserID], buffer)

			// 更新itemUp (物品因子更新公式的分子部分: Σ(r_ui * p_u))
			// bug innerItemID -> innerUserID
			copy(buffer, N.userFactor[innerUserID])
			// buffer = rating * p_u
			mulConst(rating, buffer)
			// itemUp[i] += rating * p_u
			floats.Add(itemUp[innerItemID], buffer)

			// 更新itemDown (物品因子更新公式的分母部分: Σ(p_u^T * q_i * p_u) + λ * q_i)
			copy(buffer, N.userFactor[innerUserID])
			// buffer = (p_u^T * q_i) * p_u
			mulConst(prediction, buffer)
			// itemDown[i] += prediction * p_u
			floats.Add(itemDown[innerItemID], buffer)

			copy(buffer, N.itemFactor[innerItemID])
			// buffer = reg * q_i (正则化项)
			mulConst(reg, buffer)
			// itemDown[i] += reg * q_i
			floats.Add(itemDown[innerItemID], buffer)

		}
		// 更新用户因子: p_u ← p_u ⊙ (userUp[u] / userDown[u])
		// 其中 ⊙ 表示元素级乘法，/ 表示元素级除法
		for u := range N.userFactor {
			copy(buffer, userUp[u])
			// buffer = userUp[u] / userDown[u]
			floats.Div(buffer, userDown[u])
			floats.Mul(N.userFactor[u], buffer)
		}
		// 更新物品因子: q_i ← q_i ⊙ (itemUp[i] / itemDown[i])
		for i := range N.itemFactor {
			copy(buffer, itemUp[i])
			// buffer = itemUp[i] / itemDown[i]
			floats.Div(itemUp[i], itemDown[i])
			// q_i *= buffer (乘法更新)
			floats.Mul(N.itemFactor[i], buffer)
		}
	}
}

func NewNMF() *NMF {
	return new(NMF)
}
