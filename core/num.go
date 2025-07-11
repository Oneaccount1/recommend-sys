package core

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
	"math"
	"math/rand/v2"
)

func abs(dst []float64) {
	for i := 0; i < len(dst); i++ {
		dst[i] = math.Abs(dst[i])
	}
}
func Copy(dst, src []float64) []float64 {
	copy(dst, src)
	return dst
}
func MulConst(c float64, dst []float64) []float64 {
	for i := 0; i < len(dst); i++ {
		dst[i] *= c
	}
	return dst
}

func DivConst(c float64, dst []float64) []float64 {
	for i := 0; i < len(dst); i++ {
		dst[i] /= c
	}
	return dst
}

func concatenate(arrs ...[]int) []int {
	// Sum lengths
	total := 0
	for _, arr := range arrs {
		total += len(arr)
	}
	// concatenate
	ret := make([]int, total)
	pos := 0
	for _, arr := range arrs {
		for _, val := range arr {
			ret[pos] = val
			pos++
		}
	}
	return ret
}

func NewNormalVector(size int, mean float64, stdDev float64) []float64 {
	ret := make([]float64, size)
	for i := 0; i < len(ret); i++ {
		ret[i] = rand.NormFloat64()*stdDev + mean
	}
	return ret
}

func NewUniformVector(size int, low float64, high float64) []float64 {
	ret := make([]float64, size)
	scale := high - low
	for i := 0; i < len(ret); i++ {
		ret[i] = rand.Float64()*scale + low
	}
	return ret
}
func unique(a []int) Set {
	set := make(map[int]interface{})
	for _, val := range a {
		set[val] = nil
	}
	return set
}

func selectFloat(a []float64, indices []int) []float64 {
	ret := make([]float64, len(indices))
	for id, index := range indices {
		ret[id] = a[index]
	}
	return ret
}

func selectInt(a []int, indices []int) []int {
	ret := make([]int, len(indices))
	for id, index := range ret {
		ret[id] = a[index]
	}
	return ret
}

// Metrics 矩阵
type Metrics func([]float64, []float64) float64

func RootMeanSquareError(prediction []float64, truth []float64) float64 {
	tmp := make([]float64, len(prediction))
	floats.SubTo(tmp, prediction, truth)
	floats.Mul(tmp, tmp)
	return math.Sqrt(stat.Mean(tmp, nil))
}
func MeanAbsoluteError(prediction []float64, truth []float64) float64 {
	tmp := make([]float64, len(prediction))
	floats.SubTo(tmp, prediction, truth)
	abs(tmp)
	return stat.Mean(tmp, nil)
}
