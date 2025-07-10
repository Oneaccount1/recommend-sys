package main

import (
	"fmt"
	"math/rand"
	"recommend-sys/core/algo"
	"recommend-sys/cv"
	"recommend-sys/data"
	"time"
)

func main() {
	// 创建随机数种子
	rand.New(rand.NewSource(100))

	// 创建算法实例
	random := algo.NewRandomRecommend()
	baseline := algo.NewBaseLineRecommender()
	svd := algo.NewSVDRecommend()
	svdpp := algo.NewSVDPPRecommend()
	set := data.LoadDataSet()
	var start time.Time

	start = time.Now()
	fmt.Println(cv.CrossValidate(random, set, []string{"RMSE"}, 5))
	fmt.Println(time.Since(start))

	start = time.Now()
	fmt.Println(cv.CrossValidate(baseline, set, []string{"RMSE"}, 5))
	fmt.Println(time.Since(start))

	start = time.Now()
	fmt.Println(cv.CrossValidate(svd, set, []string{"RMSE"}, 5))
	fmt.Println(time.Since(start))

	start = time.Now()
	fmt.Println(cv.CrossValidate(svdpp, set, []string{"RMSE"}, 5))
	fmt.Println(time.Since(start))

}
