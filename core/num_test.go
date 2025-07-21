package core

import (
	"fmt"
	"gonum.org/v1/gonum/stat"
	"math"
	"testing"
)

func Test_unique(t *testing.T) {
	arr := []int{1, 2, 3, 4, 5, 3, 1, 2, 4, 5, 3, 2}
	mp := unique(arr)
	for id, val := range mp {
		fmt.Println(id, val)
	}
}

func TestNewNormalVector(t *testing.T) {
	a := newNormalVector(1000, 1, 2)
	mean := stat.Mean(a, nil)
	stdDev := stat.StdDev(a, nil)
	if math.Abs(mean-1) > 0.2 {
		t.Fatalf("Mean(%.4f) doesn't match %.4f", mean, 1.0)
	} else if math.Abs(stdDev-2) > 0.2 {
		t.Fatalf("Std(%.4f) doesn't match %.4f", stdDev, 2.0)
	}
}

func Test_newUniformVectorInt(t *testing.T) {
	size, low, high := 10, 5, 10
	vectorInt := newUniformVectorInt(size, low, high)
	for i := range vectorInt {
		if vectorInt[i] > high || vectorInt[i] < low {
			t.Fatalf("vectorInt[%d] is not in scale %d and %d ", i, low, high)
		}
	}

}
