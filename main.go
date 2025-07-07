package main

import (
	"math/rand"
	"recommend-sys/algo"
	"recommend-sys/cv"
	"recommend-sys/data"
)

func main() {
	// 创建随机数种子
	rand.New(rand.NewSource(100))

	// 创建算法实例
	random := &algo.Random{}

	set := data.LoadDataSet()
	cv.CrossValidate(random, set, 5)

}
