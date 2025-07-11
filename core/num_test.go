package core

import (
	"fmt"
	"testing"
)

func Test_unique(t *testing.T) {
	arr := []int{1, 2, 3, 4, 5, 3, 1, 2, 4, 5, 3, 2}
	mp := unique(arr)
	for id, val := range mp {
		fmt.Println(id, val)
	}
}
