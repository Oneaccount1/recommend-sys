package core

import (
	"gonum.org/v1/gonum/stat"
	"math"
	"reflect"
)

// ParameterGrid 实际上就是一个二维数组
type ParameterGrid map[string][]interface{}

type CrossValidateResult struct {
	Trains []float64
	Tests  []float64
}

// CrossValidate 验证推荐算法性能
func CrossValidate(estimator Estimator, dataSet DataSet, metrics []Evaluator, cv int, seed int64,
	params Parameters, nJobs int) []CrossValidateResult {

	ret := make([]CrossValidateResult, len(metrics))
	for i := 0; i < len(ret); i++ {
		ret[i].Trains = make([]float64, cv)
		ret[i].Tests = make([]float64, cv)
	}
	// 分割测试集合
	trainFolds, testFolds := dataSet.KFold(cv, seed)
	parallel(cv, nJobs, func(begin, end int) {
		cp := reflect.New(reflect.TypeOf(estimator).Elem()).Interface().(Estimator)
		Copy(cp, estimator)
		for i := begin; i < end; i++ {
			trainFold := trainFolds[i]
			testFold := testFolds[i]
			cp.SetParams(params)
			cp.Fit(trainFold)
			// Evaluate on test set
			testRatings := testFold.Ratings
			testPredictions := testFold.Predict(cp)
			for j := 0; j < len(metrics); j++ {
				ret[j].Tests[i] = metrics[j](testPredictions, testRatings)
			}
			// Evaluate on train set
		}

	})

	//for i := 0; i < cv; i++ {
	//	// 训练
	//	trainFold := trainFolds[i]
	//	testFold := testFolds[i]
	//	estimator.SetParams(params)
	//	estimator.Fit(trainFold)
	//	// Evaluate on train set
	//	trainRatings := trainFold.Ratings
	//	trainPrediction := trainFold.Predict(estimator)
	//	for j := 0; j < len(ret); j++ {
	//		ret[j].Trains[i] = metrics[j](trainPrediction, trainRatings)
	//	}
	//	// Evaluate on test set
	//	testRatings := testFold.Ratings
	//	testPredictions := testFold.Predict(estimator)
	//	for j := 0; j < len(ret); j++ {
	//		ret[j].Tests[i] = metrics[j](testPredictions, testRatings)
	//	}
	//}
	return ret
}

type GridSearchResult struct {
	BestScore  float64
	BestParams Parameters
	BestIndex  int
	CVResult   []CrossValidateResult
	AllParams  []Parameters
}

// GridSearchCV Tune algorithm parameters with GridSearchCV
func GridSearchCV(algo Estimator, dataSet DataSet, paramGrid ParameterGrid,
	evaluators []Evaluator, cv int, seed int64) []GridSearchResult {
	// 获取参数名字和长度
	params := make([]string, 0, len(paramGrid))
	count := 1
	for param, values := range paramGrid {
		params = append(params, param)
		// 计算每一种参数可能的组合
		count *= len(values)
	}
	// 创建网格搜索结果
	// 每种验证方式产生一种结果
	results := make([]GridSearchResult, len(evaluators))

	// 初始化
	for i := range results {
		results[i] = GridSearchResult{}
		results[i].BestScore = math.Inf(1)
		results[i].CVResult = make([]CrossValidateResult, 0, count)
		results[i].AllParams = make([]Parameters, 0, count)
	}

	// DFS
	var dfs func(deep int, options Parameters)

	dfs = func(deep int, options Parameters) {
		// 当deep == len(params) 时候，说明已经遍历完所有参数
		if deep == len(params) {
			// Cross validate
			cvResult := CrossValidate(algo, dataSet, evaluators, cv, seed, options)

			for i := range cvResult {
				results[i].CVResult = append(results[i].CVResult, cvResult[i])
				// 复制当前层的参数
				results[i].AllParams = append(results[i].AllParams, options.Copy())
				// 计算测试集平均分
				score := stat.Mean(cvResult[i].Tests, nil)
				if score < results[i].BestScore {
					results[i].BestScore = score
					results[i].BestParams = options.Copy()
					results[i].BestIndex = len(results[i].AllParams) - 1
				}
			}
		} else {
			// 选取下一个参数
			param := params[deep]
			values := paramGrid[param]
			// 遍历下一个参数的可能选值
			for _, val := range values {
				options[param] = val
				dfs(deep+1, options)
			}
		}
	}

	options := make(map[string]interface{})
	dfs(0, options)
	return results
}

// NewAUC 创建AUC (Area Under the ROC Curve) 评估器
// AUC是推荐系统中常用的评估指标，用于衡量模型区分正负样本的能力
//
// 数学原理：
// AUC = (1/|U|) * Σ_{u∈U} AUC_u
// 其中 AUC_u = (1/(|I_u^+| * |I_u^-|)) * Σ_{i∈I_u^+} Σ_{j∈I_u^-} I(r̂_{ui} > r̂_{uj})
//
// 符号说明：
// - U: 用户集合
// - I_u^+: 用户u在测试集中评分的物品集合（正样本）
// - I_u^-: 用户u在完整数据集中未评分的物品集合（负样本）
// - r̂_{ui}: 模型预测用户u对物品i的评分
// - I(·): 指示函数，条件为真时返回1，否则返回0
//
// 参数：
// - fullSet: 完整的数据集，用于确定负样本（用户未评分的物品）
//
// 返回值：
// - Evaluator: 评估函数，接受估计器和测试集，返回AUC值
func NewAUC(fullSet DataSet) Evaluator {
	return func(estimator Estimator, testSet DataSet) float64 {
		// 将数据集转换为TrainSet格式，便于处理用户-物品映射
		full := NewTrainSet(fullSet)
		test := NewTrainSet(testSet)

		// 累计所有用户的AUC值和用户计数
		userAUCSum, userCount := 0.0, 0.0

		// 遍历测试集中的每个用户
		for innerUserIDTest, testRatings := range test.UserRatings() {
			// 获取用户的外部ID（原始用户ID）
			userID := test.outerUserIDs[innerUserIDTest]

			// 在完整数据集中查找该用户的内部ID
			innerUserIDFull := full.ConvertUserID(userID)

			// 如果用户在完整数据集中不存在，跳过该用户
			if innerUserIDFull == newID {
				continue
			}

			// 构建用户在完整数据集中已评分物品的映射表
			// key: 物品的外部ID, value: 评分值
			fullRatedItems := make(map[int]float64)
			for _, rating := range full.UserRatings()[innerUserIDFull] {
				// 修复错误：应该使用outerItemIDs而不是outerUserIDs
				itemID := full.outerItemIDs[rating.ID]
				fullRatedItems[itemID] = rating.Rating
			}

			// 计算当前用户的AUC值
			correctPairs, totalPairs := 0.0, 0.0

			// 遍历测试集中该用户的每个评分物品（正样本）
			for _, testRating := range testRatings {
				// 修复错误：应该使用outerItemIDs而不是outerUserIDs
				positiveItemID := test.outerItemIDs[testRating.ID]

				// 遍历完整数据集中的所有物品，寻找负样本
				for j := 0; j < full.ItemCount; j++ {
					negativeItemID := full.outerItemIDs[j]

					// 如果该物品在完整数据集中未被用户评分，则作为负样本
					if _, exists := fullRatedItems[negativeItemID]; !exists {
						// 比较模型对正样本和负样本的预测评分
						positivePrediction := estimator.Predict(userID, positiveItemID)
						negativePrediction := estimator.Predict(userID, negativeItemID)

						// 如果正样本预测评分高于负样本，则为正确排序
						if positivePrediction > negativePrediction {
							correctPairs++
						}
						totalPairs++
					}
				}
			}

			// 计算当前用户的AUC值并累加
			// AUC_u = correctPairs / totalPairs
			if totalPairs > 0 {
				userAUCSum += correctPairs / totalPairs
				userCount++
			}
		}

		// 返回所有用户AUC值的平均值
		// 如果没有有效用户，返回0.5（随机猜测的AUC值）
		if userCount > 0 {
			return userAUCSum / userCount
		}
		return 0.5
	}
}
