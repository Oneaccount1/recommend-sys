package core

import (
	"math"
)

type Sim func(SortedIdRatings, SortedIdRatings) float64

// Cosine 余弦相似度
func Cosine(a SortedIdRatings, b SortedIdRatings) float64 {
	m, n, l := .0, .0, .0
	ptr := 0
	for _, ir := range a.data {
		for ptr < len(b.data) && b.data[ptr].ID < ir.ID {
			ptr++
		}
		if ptr < len(b.data) && b.data[ptr].ID == ir.ID {
			jr := b.data[ptr]
			m += ir.Rating * ir.Rating
			n += jr.Rating * jr.Rating
			l += ir.Rating * jr.Rating
		}
	}
	return l / (math.Sqrt(m) * math.Sqrt(n))
}

// MSD 均方差相似度
func MSD(a SortedIdRatings, b SortedIdRatings) float64 {
	count, sum, ptr := 0.0, 0.0, 0

	for _, ir := range a.data {
		for ptr < len(b.data) && b.data[ptr].ID < ir.ID {
			ptr++
		}
		if ptr < len(b.data) && b.data[ptr].ID == ir.ID {
			jr := b.data[ptr]
			sum += (ir.Rating - jr.Rating) * (ir.Rating - jr.Rating)
			count++
		}

	}

	return 1.0 / (sum/count + 1.0)
}

// Pearson 皮尔逊相似度
func Pearson(a SortedIdRatings, b SortedIdRatings) float64 {
	// A 平均值
	count, sum := .0, .0
	for _, ir := range a.data {
		sum += ir.Rating
		count += 1
	}
	meanA := sum / count

	// B 平均值
	count, sum = .0, .0
	for _, ir := range b.data {
		sum += ir.Rating
		count += 1
	}
	meanB := sum / count

	// 去中心化的余弦相似度
	m, n, l := .0, .0, .0
	ptr := 0
	for _, ir := range a.data {
		for ptr < len(b.data) && b.data[ptr].ID < ir.ID {
			ptr++
		}
		if ptr < len(b.data) && b.data[ptr].ID == ir.ID {
			jr := b.data[ptr]
			ratingA := ir.Rating - meanA
			ratingB := jr.Rating - meanB
			m += ratingA * ratingA
			n += ratingB * ratingB
			l += ratingA * ratingB
		}
	}
	return l / (math.Sqrt(m) * math.Sqrt(n))
}
