package core

import (
	"bufio"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

type Set map[int]interface{}

type TrainSet struct {
	interactionRatings []float64
	interactionUsers   []int
	interactionItems   []int
	userRatings        map[int]map[int]float64
	itemRatings        map[int]map[int]float64
}

func (set *TrainSet) Length() int {

	return len(set.interactionRatings)
}
func (set *TrainSet) Interactions() ([]int, []int, []float64) {
	return set.interactionUsers, set.interactionItems, set.interactionRatings
}
func (set *TrainSet) Users() Set {
	return unique(set.interactionUsers)
}
func (set *TrainSet) Items() Set {
	return unique(set.interactionItems)
}
func (set *TrainSet) RatingRange() (float64, float64) {
	return floats.Min(set.interactionRatings), floats.Max(set.interactionRatings)
}
func (set *TrainSet) GlobalMean() float64 {
	return stat.Mean(set.interactionRatings, nil)
}

func (set *TrainSet) UserRatings() map[int]map[int]float64 {
	if set.userRatings == nil {
		set.userRatings = make(map[int]map[int]float64)
		users, items, ratings := set.Interactions()
		for i := 0; i < len(users); i++ {
			userID := users[i]
			itemID := items[i]
			// 首次需要初始化
			if _, exist := set.userRatings[userID]; !exist {
				set.userRatings[userID] = make(map[int]float64)
			}
			// 插入物品信息
			set.userRatings[userID][itemID] = ratings[i]
		}
	}
	return set.userRatings
}

func (set *TrainSet) ItemRatings() map[int]map[int]float64 {
	if set.itemRatings == nil {
		set.itemRatings = make(map[int]map[int]float64)
		users, items, ratings := set.Interactions()
		for i := 0; i < len(users); i++ {
			userID := users[i]
			itemID := items[i]
			if _, exist := set.itemRatings[itemID]; !exist {
				set.itemRatings[itemID] = make(map[int]float64)
			}
			// 插入物品
			set.itemRatings[itemID][userID] = ratings[i]
		}
	}
	return set.itemRatings
}
func (set *TrainSet) KFold(k int, seed int64) ([]TrainSet, []TrainSet) {
	trainFolds := make([]TrainSet, k)
	testFolds := make([]TrainSet, k)
	rand.New(rand.NewSource(seed))
	perm := rand.Perm(set.Length())
	foldSize := set.Length() / k
	begin, end := 0, 0
	// todo 数据集划分
	for i := 0; i < k; i++ {
		end += foldSize
		if i < set.Length()%k {
			end++
		}
		// Test set
		testIndex := perm[begin:end]
		testFolds[i].interactionUsers = selectInt(set.interactionUsers, testIndex)
		testFolds[i].interactionItems = selectInt(set.interactionItems, testIndex)
		testFolds[i].interactionRatings = selectFloat(set.interactionRatings, testIndex)
		// Train set
		trainIndex := concatenate(perm[0:begin], perm[end:set.Length()])
		trainFolds[i].interactionUsers = selectInt(set.interactionUsers, trainIndex)
		trainFolds[i].interactionItems = selectInt(set.interactionItems, trainIndex)
		trainFolds[i].interactionRatings = selectFloat(set.interactionRatings, trainIndex)
		begin = end
	}
	return trainFolds, testFolds
}

func LoadDataFromBuiltIn(dataSetName string) TrainSet {
	// Extract data set information
	dataSet, exist := buildInDataSet[dataSetName]
	if !exist {
		log.Fatal("no such data set", dataSetName)
	}
	const dataFolder = "data"
	const tempFolder = "temp"
	dataFileName := filepath.Join(dataFolder, dataSet.path)
	if _, err := os.Stat(dataFileName); os.IsNotExist(err) {
		zipFileName, _ := DownloadFromUrl(dataSet.url, tempFolder)
		Unzip(zipFileName, dataFolder)
	}
	return LoadDataFromFile(dataFileName, dataSet.sep)
}

func LoadDataFromFile(fileName string, sep string) TrainSet {
	set := TrainSet{}
	set.interactionUsers = make([]int, 0)
	set.interactionItems = make([]int, 0)
	set.interactionRatings = make([]float64, 0)

	// 打开文件
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}

	defer file.Close()

	// Read CSV file
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, "\t")
		user, _ := strconv.Atoi(fields[0])
		item, _ := strconv.Atoi(fields[1])
		ratings, _ := strconv.Atoi(fields[2])
		//
		set.interactionUsers = append(set.interactionUsers, user)
		set.interactionItems = append(set.interactionItems, item)
		set.interactionRatings = append(set.interactionRatings, float64(ratings))
	}
	return set
}
