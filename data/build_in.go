package data

import (
	"bufio"
	"math/rand"
	"os"

	"github.com/go-gota/gota/dataframe"
)

type Set struct {
	Interactions dataframe.DataFrame
	NRatings     int
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
