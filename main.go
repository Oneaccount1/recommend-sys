package main

import "fmt"

type stu struct {
	Name string
}

func main() {
	a := []int{1, 2, 3, 4, 5}
	var b []int
	b = make([]int, len(a))
	copy(b, a)
	a[0] = 10
	for _, val := range a {
		fmt.Println(val)
	}
	for _, val := range b {
		fmt.Println(val)
	}
}
