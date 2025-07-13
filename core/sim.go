package core

import "math"

type Sim func([]float64, []float64) float64

// Cosine 余弦相似度
func Cosine(a []float64, b []float64) float64 {
	m, n, l := .0, .0, .0
	for i := range a {
		m += a[i] * a[i]
		n += b[i] * b[i]
		l += a[i] * b[i]
	}
	return l / (math.Sqrt(m) * math.Sqrt(n))
}

// MSD 均方差相似度
func MSD(a []float64, b []float64) float64 {
	count := .0
	sum := .0
	for i := range a {
		if !(math.IsNaN(a[i]) || math.IsNaN(b[i])) {
			sum += (a[i] - b[i]) * (a[i] - b[i])
			count++
		}

	}
	return 1.0 / (sum/count + 1.0)
}

// Pearson 皮尔逊相似度
func Pearson(a []float64, b []float64) float64 {
	// A 平均值
	count, sum := .0, .0
	for i := range a {
		sum += a[i]
		count += 1
	}
	meanA := sum / count

	// B 平均值
	count, sum = .0, .0
	for i := range b {
		sum += b[i]
		count += 1
	}
	meanB := sum / count

	//// 去中心化的余弦相似度
	m, n, l := .0, .0, .0
	for i := range a {
		if !(math.IsNaN(a[i]) || math.IsNaN(b[i])) {
			ratingA := a[i] - meanA
			ratingB := b[i] - meanB
			m += ratingA * ratingA
			n += ratingB * ratingB
			l += ratingA * ratingB
		}
	}
	return l / (math.Sqrt(m) * math.Sqrt(n))
}
