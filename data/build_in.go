package data

import (
	"bufio"
	"gonum.org/v1/gonum/floats"
	"math/rand"
	"os"

	"github.com/go-gota/gota/dataframe"
)

type Set struct {
	Interactions dataframe.DataFrame
	NRatings     int

	users []int
	items []int

	userMap map[int]int
	itemMap map[int]int
}

func Unique(a []int) []int {
	memo := make(map[int]bool)

	ret := make([]int, 0, len(a))

	for _, v := range a {
		if _, exist := memo[v]; !exist {
			memo[v] = true
			ret = append(ret, v)
		}
	}

	return ret
}

func NewDataSet(df dataframe.DataFrame) Set {
	set := Set{}
	set.Interactions = df
	set.NRatings = df.Nrow()

	// 创建用户表
	users, _ := set.Interactions.Col("X0").Int()
	set.users = Unique(users)
	set.userMap = make(map[int]int)

	for innerUserID, userID := range set.users {
		set.userMap[userID] = innerUserID
	}
	// 创建物品表
	items, _ := set.Interactions.Col("X1").Int()
	set.items = Unique(items)
	set.itemMap = make(map[int]int)
	for innerItemID, itemID := range set.items {
		set.itemMap[itemID] = innerItemID
	}
	return set

}

func (s *Set) KFold(k int) ([]dataframe.DataFrame, []dataframe.DataFrame) {
	testSet := make([]dataframe.DataFrame, k)
	trainSet := make([]dataframe.DataFrame, k)

	perm := rand.Perm(s.NRatings)

	foldSize := s.NRatings / k

	for i := 0; i < s.NRatings; i += foldSize {

		j := i + foldSize

		if j >= s.NRatings {
			j = s.NRatings
		}

		testSet[i/foldSize] = s.Interactions.Subset(perm[i:j])
		trainSet[i/foldSize] = s.Interactions.Subset(append(perm[0:i], perm[j:s.NRatings]...))

	}

	return trainSet, testSet
}

func (s *Set) AllRatings() []float64 {
	// 通过gota的DataFrame的Col方法获取"X2"列(评分列)，并转换为float64切片
	return s.Interactions.Col("X2").Float()
}
func (s *Set) NRow() int {
	return s.Interactions.Nrow()
}
func (s *Set) AllInteraction() ([]int, []int, []float64) {
	users, _ := s.Interactions.Col("X0").Int()
	items, _ := s.Interactions.Col("X1").Int()
	ratings := s.Interactions.Col("X2").Float()

	return users, items, ratings
}

func (s *Set) AllUsers() []int {
	return s.users
}

func (s *Set) AllItems() []int {
	return s.items
}
func (s *Set) RatingRange() (float64, float64) {
	ratings := s.AllRatings()

	return floats.Min(ratings), floats.Max(ratings)
}

func LoadDataSet() Set {
	set := Set{}

	set.Interactions = readCSV("data/ml-100k/u.data")

	set.NRatings = set.Interactions.Nrow()

	return set
}
func readCSV(fileName string) dataframe.DataFrame {
	file, _ := os.Open(fileName)
	df := dataframe.ReadCSV(bufio.NewReader(file),
		dataframe.WithDelimiter('\t'),
		dataframe.HasHeader(false),
	)
	return df
}
