package core

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"math"
	"runtime"
	"sync"
)

// The famous SVD algorithm, as popularized by Simon Funk during the
// Netflix Prize. The prediction \hat{r}_{ui} is set as:
//
//	\hat{r}_{ui} = μ + b_u + b_i + q_i^Tp_u
//
// If user u is unknown, then the Bias b_u and the factors p_u are
// assumed to be zero. The same applies for item i with b_i and q_i.
type SVD struct {
	Base
	userFactor [][]float64
	itemFactor [][]float64
	userBias   []float64
	itemBias   []float64
	globalBias float64
	trainSet   TrainSet
}

func NewSVD(params Parameters) *SVD {
	svd := new(SVD)
	svd.Params = params
	return svd
}
func (s *SVD) Predict(userID, itemID int) float64 {
	innerUserID := s.trainSet.ConvertUserID(userID)
	innerItemID := s.trainSet.ConvertItemID(itemID)
	ret := s.globalBias
	// +b_u
	if innerUserID != newID {
		ret += s.userBias[innerUserID]
	}
	// +b_i
	if innerItemID != newID {
		ret += s.itemBias[innerItemID]
	}
	// +q_i^Tp_u
	if !(innerItemID == newID || innerUserID == newID) {
		userFactor := s.userFactor[innerUserID]
		itemFactor := s.itemFactor[innerItemID]
		ret += floats.Dot(userFactor, itemFactor)
	}
	return ret
}

// Fit a SVD model.
// Parameters:
//
//	 reg 		- The regularization parameter of the cost function that is
//				  optimized. Default is 0.02.
//	 lr 		- The learning rate of SGD. Default is 0.005.
//	 nFactors	- The number of latent factors. Default is 100.
//	 nEpochs	- The number of iteration of the SGD procedure. Default is 20.
//	 initMean	- The Means of initial random latent factors. Default is 0.
//	 initStdDev	- The standard deviation of initial random latent factors. Default is 0.1.
func (s *SVD) Fit(trainData TrainSet) {
	// Setup parameters
	nFactors := s.Params.GetInt("nFactors", 100)
	nEpochs := s.Params.GetInt("nEpochs", 20)
	lr := s.Params.GetFloat64("lr", 0.005)
	reg := s.Params.GetFloat64("reg", 0.02)
	//biased := options.GetBool("biased", true)
	initMean := s.Params.GetFloat64("initMean", 0)
	initStdDev := s.Params.GetFloat64("initStdDev", 0.1)
	// 初始化参数
	s.trainSet = trainData
	s.userFactor = make([][]float64, s.trainSet.UserCount)
	s.itemFactor = make([][]float64, s.trainSet.ItemCount)

	s.itemBias = make([]float64, s.trainSet.ItemCount)
	s.userBias = make([]float64, s.trainSet.UserCount)

	for i := range s.userFactor {
		s.userFactor[i] = newNormalVector(nFactors, initMean, initStdDev)
	}
	for i := range s.itemFactor {
		s.itemFactor[i] = newNormalVector(nFactors, initMean, initStdDev)
	}

	// 创建缓存
	a := make([]float64, nFactors)
	b := make([]float64, nFactors)

	// 随机梯度下降
	for epoch := 0; epoch < nEpochs; epoch++ {
		for i := 0; i < trainData.Length(); i++ {
			userID, itemID, rating := trainData.Index(i)
			innerUserID := trainData.ConvertUserID(userID)
			innerItemID := trainData.ConvertItemID(itemID)
			userBias := s.userBias[innerUserID]
			itemBias := s.itemBias[innerItemID]
			userFactor := s.userFactor[innerUserID]
			itemFactor := s.itemFactor[innerItemID]
			// 计算差值
			diff := s.Predict(userID, itemID) - rating

			// 计算各个参数的梯度
			gradGlobalBias := diff
			s.globalBias -= lr * gradGlobalBias

			gradUserBias := diff + reg*userBias
			s.userBias[innerUserID] -= lr * gradUserBias

			gradItemBias := diff + reg*itemBias
			s.itemBias[innerItemID] -= lr * gradItemBias
			// update user latent factor
			copy(a, itemFactor)
			mulConst(diff, a)
			copy(b, userFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(s.userFactor[innerUserID], a)

			copy(a, userFactor)
			mulConst(diff, a)
			copy(b, itemFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(s.itemFactor[innerItemID], a)
		}
	}

}

type NMF struct {
	Base
	userFactor [][]float64 // p_u
	itemFactor [][]float64 // q_i
}

func (N *NMF) Predict(userId int, itemId int) float64 {
	innerUserID := N.Data.ConvertUserID(userId)
	innerItemID := N.Data.ConvertItemID(itemId)
	if innerUserID != newID && innerItemID != newID {
		return floats.Dot(N.userFactor[innerUserID], N.itemFactor[innerItemID])
	}
	return 0
}

// Fit a NMF model.
// Parameters:
//
//	 reg 		- The regularization parameter of the cost function that is
//				  optimized. Default is 0.06.
//	 nFactors	- The number of latent factors. Default is 15.
//	 nEpochs	- The number of iteration of the SGD procedure. Default is 50.
//	 initLow	- The lower bound of initial random latent factor. Default is 0.
//	 initHigh	- The upper bound of initial random latent factor. Default is 1.
func (N *NMF) Fit(trainSet TrainSet) {
	nFactors := N.Params.GetInt("nFactors", 15)
	nEpochs := N.Params.GetInt("nEpochs", 50)
	initLow := N.Params.GetFloat64("initLow", 0)
	initHigh := N.Params.GetFloat64("initHigh", 1)
	reg := N.Params.GetFloat64("reg", 0.06)
	//lr := options.GetFloat64("lr", 0.005)
	N.Data = trainSet
	// 初始化参数
	N.userFactor = newUniformMatrix(trainSet.UserCount, nFactors, initLow, initHigh)
	N.itemFactor = newUniformMatrix(trainSet.ItemCount, nFactors, initLow, initHigh)

	// 创建Buffer
	buffer := make([]float64, nFactors)
	// 创建用于存储乘法更新规则分子和分母的中间矩阵
	userUp := newZeroMatrix(trainSet.UserCount, nFactors)
	userDown := newZeroMatrix(trainSet.UserCount, nFactors)
	itemUp := newZeroMatrix(trainSet.ItemCount, nFactors)
	itemDown := newZeroMatrix(trainSet.ItemCount, nFactors)

	for epoch := 0; epoch < nEpochs; epoch++ {
		// 重置中间矩阵
		resetZeroMatrix(userUp)
		resetZeroMatrix(userDown)
		resetZeroMatrix(itemUp)
		resetZeroMatrix(itemDown)

		// 计算中间矩阵
		for i := 0; i < trainSet.Length(); i++ {
			userID, itemID, rating := trainSet.Users[i], trainSet.Items[i], trainSet.Ratings[i]
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

func NewNMF(params Parameters) *NMF {
	nmf := new(NMF)
	nmf.Params = params
	return nmf
}

type SVDPP struct {
	Base
	UserRatings [][]IDRating
	UserFactor  [][]float64
	ItemFactor  [][]float64
	ImplFactor  [][]float64 //y_i
	//cacheFactor map[int][]float64
	UserBias   []float64
	ItemBias   []float64
	GlobalBias float64
}

func (pp *SVDPP) ensembleImplFactors(innerUserID int) []float64 {
	nFactors := pp.Params.GetInt("nFactors", 20)
	emImpFactor := make([]float64, nFactors)
	count := 0
	// 用户交互物品历史存在
	for _, ir := range pp.UserRatings[innerUserID] {
		floats.Add(emImpFactor, pp.ImplFactor[ir.ID])
		count++
	}
	divConst(math.Sqrt(float64(count)), emImpFactor)
	return emImpFactor
}

func (pp *SVDPP) internalPredict(userID, itemID int) (float64, []float64) {

	// convert to inner id

	innerUserID := pp.Data.ConvertUserID(userID)
	innerItemID := pp.Data.ConvertItemID(itemID)
	ret := pp.GlobalBias
	if innerUserID != newID {
		ret += pp.UserBias[innerUserID]
	}

	if innerItemID != newID {
		ret += pp.ItemBias[innerItemID]
	}
	if innerItemID != newID && innerUserID != newID {
		userFactor := pp.UserFactor[innerUserID]
		itemFactor := pp.ItemFactor[innerItemID]
		emImpFactor := pp.ensembleImplFactors(innerUserID)
		tmp := make([]float64, len(itemFactor))
		floats.Add(tmp, userFactor)
		floats.Add(tmp, emImpFactor)
		ret += floats.Dot(tmp, itemFactor)
		return ret, emImpFactor
	}
	return ret, []float64{}

}
func (pp *SVDPP) Predict(userID, itemID int) float64 {
	predict, _ := pp.internalPredict(userID, itemID)
	return predict
}

func (pp *SVDPP) Fit(trainData TrainSet) {
	nFactors := pp.Params.GetInt("nFactors", 20)
	nEpochs := pp.Params.GetInt("nEpochs", 20)
	lr := pp.Params.GetFloat64("lr", 0.007)
	reg := pp.Params.GetFloat64("reg", 0.02)
	initMean := pp.Params.GetFloat64("initMean", 0)
	initStdDev := pp.Params.GetFloat64("initStdDev", 0.1)
	// nJob并发执行
	nJobs := pp.Params.GetInt("nJobs", runtime.NumCPU())
	// 初始化参数
	pp.Data = trainData
	pp.UserBias = make([]float64, trainData.UserCount)
	pp.ItemBias = make([]float64, trainData.ItemCount)
	pp.UserFactor = make([][]float64, trainData.UserCount)
	pp.ItemFactor = make([][]float64, trainData.ItemCount)
	pp.ImplFactor = make([][]float64, trainData.ItemCount)
	//pp.cacheFactor = make(map[int][]float64)

	for innerUserID := range pp.UserBias {
		pp.UserFactor[innerUserID] = newNormalVector(nFactors, initMean, initStdDev)
	}
	for innerItemID := range pp.ItemBias {
		pp.ItemFactor[innerItemID] = newNormalVector(nFactors, initMean, initStdDev)
		pp.ImplFactor[innerItemID] = newNormalVector(nFactors, initMean, initStdDev)
	}
	// 创建用户历史物品
	pp.UserRatings = trainData.UserRatings()

	// 创建缓存
	a := make([]float64, nFactors)
	b := make([]float64, nFactors)

	// 随即梯度下降算法
	// 系数常数已经保存在学习率和正则化系数中
	for epoch := 0; epoch < nEpochs; epoch++ {
		fmt.Printf("第 %d 轮\n", epoch)
		for i := 0; i < trainData.Length(); i++ {
			userID, itemID, rating := trainData.Index(i)

			innerUserID := trainData.ConvertUserID(userID)
			innerItemID := trainData.ConvertItemID(itemID)

			userBias := pp.UserBias[innerUserID]
			itemBias := pp.ItemBias[innerItemID]
			userFactor := pp.UserFactor[innerUserID]
			itemFactor := pp.ItemFactor[innerItemID]
			// 计算差值
			pred, emImpFactor := pp.internalPredict(userID, itemID)
			diff := pred - rating
			// 更新全局偏置
			gradGlobalBias := diff
			pp.GlobalBias -= lr * gradGlobalBias

			// 更新 User 偏置
			gradUserBias := diff + reg*userBias
			pp.UserBias[innerUserID] -= lr * gradUserBias

			// item  偏置
			gradItemBias := diff + reg*itemBias
			pp.ItemBias[innerItemID] -= lr * gradItemBias

			// user 潜在因子
			copy(a, itemFactor)
			mulConst(diff, a)
			copy(b, userFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(pp.UserFactor[innerUserID], a)

			// item 潜在因子
			copy(a, userFactor)
			if len(emImpFactor) > 0 {
				floats.Add(a, emImpFactor)
			}
			mulConst(diff, a)
			copy(b, itemFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(pp.ItemFactor[innerItemID], a)
			// todo 并行计算
			// 隐因子
			nRating := len(pp.UserRatings[innerUserID])
			var wg sync.WaitGroup
			wg.Add(nJobs)
			for j := 0; j < nJobs; j++ {
				go func(jobId int) {
					low := nRating * jobId / nJobs
					high := nRating * (jobId + 1) / nJobs
					a := make([]float64, nFactors)
					b := make([]float64, nFactors)
					for i := low; i < high; i++ {
						implFactor := pp.ImplFactor[pp.UserRatings[innerUserID][i].ID]
						copy(a, itemFactor)
						mulConst(diff, a)
						divConst(math.Sqrt(float64(len(pp.UserRatings[innerUserID]))), a)
						copy(b, implFactor)
						mulConst(reg, b)
						floats.Add(a, b)
						mulConst(lr, a)
						floats.Sub(pp.ImplFactor[pp.UserRatings[innerUserID][i].ID], a)
					}
					wg.Done()
				}(j)
			}
			wg.Wait()
			//
			//for _, ir := range pp.UserRatings[innerUserID] {
			//	implFactor := pp.ImplFactor[ir.ID]
			//
			//	copy(a, itemFactor)
			//	mulConst(diff, a)
			//	divConst(math.Sqrt(float64(len(pp.UserRatings[innerUserID]))), a)
			//
			//	copy(b, implFactor)
			//	mulConst(reg, b)
			//	floats.Add(a, b)
			//	mulConst(lr, a)
			//	floats.Sub(pp.ImplFactor[ir.ID], a)
			//}

		}
	}

}

func NewSVDpp(params Parameters) *SVDPP {
	svdpp := new(SVDPP)
	svdpp.Params = params
	return svdpp
}
