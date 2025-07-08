package main

import (
	"fmt"
	"math/rand"
	"recommend-sys/algo"
	"recommend-sys/cv"
	"recommend-sys/data"
)

func main() {
	// 创建随机数种子
	rand.New(rand.NewSource(100))

	// 创建算法实例
	random := algo.NewRandomRecommend()
	baseline := algo.NewBaseLineRecommender()

	set := data.LoadDataSet()

	fmt.Println(cv.CrossValidate(random, set, []string{"RMSE"}, 5))
	fmt.Println(cv.CrossValidate(baseline, set, []string{"RMSE"}, 5))

}
