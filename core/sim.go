package core

import "math"

type Sim func(map[int]float64, map[int]float64) float64

// Cosine 余弦相似度
func Cosine(a map[int]float64, b map[int]float64) float64 {
	m, n, l := .0, .0, .0
	for id, ratingA := range a {
		if ratingB, exist := b[id]; exist {
			m += ratingA * ratingA
			n += ratingB * ratingB
			l += ratingA * ratingB
		}
	}
	return l / (math.Sqrt(m) * math.Sqrt(n))
}

// MSD 均方差相似度
func MSD(a map[int]float64, b map[int]float64) float64 {
	count := .0
	sum := .0
	for id, ratingA := range a {
		if ratingB, exist := b[id]; exist {
			sum += (ratingA - ratingB) * (ratingA - ratingB)
			count++
		}
	}
	return 1.0 / (sum/count + 1.0)
}

// Pearson 皮尔逊相似度
func Pearson(a map[int]float64, b map[int]float64) float64 {
	// A 平均值
	count, sum := .0, .0
	for _, ratingA := range a {
		sum += ratingA
		count += 1
	}
	meanA := sum / count

	// B 平均值
	count, sum = .0, .0
	for _, ratingB := range b {
		sum += ratingB
		count++
	}
	meanB := sum / count
	// 去中心化的余弦相似度
	m, n, l := .0, .0, .0
	for id, ratingA := range a {
		if ratingB, exist := b[id]; exist {
			m += (ratingA - meanA) * (ratingA - meanA)
			n += (ratingB - meanB) * (ratingB - meanB)
			l += (ratingA - meanA) * (ratingB - meanB)
		}
	}
	return l / (math.Sqrt(m) * math.Sqrt(n))
}
