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

// TrainSet 由原来的Map实现 Map[int]map[int]float   用户-物品-评分
// 转换为 map[int][int] 定位矩阵的行列位置   [][]float 评分矩阵
// 加速算法运行
type TrainSet struct {
	interactionRatings []float64
	interactionUsers   []int
	interactionItems   []int
	userCount          int
	itemCount          int

	innerUserIDs map[int]int
	innerItemIDs map[int]int

	userRatings [][]float64
	itemRatings [][]float64
}

const newID = -1

// Length 获取数据集的大小、长度
func (set *TrainSet) Length() int {
	return len(set.interactionRatings)
}

// Interactions 获取数据集的User Item Rating数据
func (set *TrainSet) Interactions() ([]int, []int, []float64) {
	return set.interactionUsers, set.interactionItems, set.interactionRatings
}

// Users 获取数据集所有User
func (set *TrainSet) Users() Set {
	return unique(set.interactionUsers)
}

// Items 获取所有数据集Item
func (set *TrainSet) Items() Set {
	return unique(set.interactionItems)
}

// RatingRange 获取数据集评分的范围
func (set *TrainSet) RatingRange() (float64, float64) {
	return floats.Min(set.interactionRatings), floats.Max(set.interactionRatings)
}

// GlobalMean 获取数据集中Rating的平均值
func (set *TrainSet) GlobalMean() float64 {
	return stat.Mean(set.interactionRatings, nil)
}

func (set *TrainSet) UserCount() int {
	return set.userCount
}
func (set *TrainSet) ItemCount() int {
	return set.itemCount
}
func (set *TrainSet) ConvertUserID(userID int) int {
	if innerUserID, exist := set.innerUserIDs[userID]; exist {
		return innerUserID
	}
	return newID
}
func (set *TrainSet) ConvertItemID(itemID int) int {
	if innerItemID, exist := set.innerItemIDs[itemID]; exist {
		return innerItemID
	}
	return newID
}
func (set *TrainSet) UserRatings() [][]float64 {
	if set.userRatings == nil {
		set.userRatings = newNanMatrix(set.userCount, set.itemCount)
		users, items, ratings := set.Interactions()
		for i := 0; i < len(users); i++ {
			innerUserID := set.ConvertUserID(users[i])
			innerItemID := set.ConvertItemID(items[i])
			set.userRatings[innerUserID][innerItemID] = ratings[i]
		}
	}
	return set.userRatings
}

func (set *TrainSet) ItemRatings() [][]float64 {
	if set.itemRatings == nil {
		set.itemRatings = newNanMatrix(set.itemCount, set.userCount)
		users, items, ratings := set.Interactions()
		for i := 0; i < len(users); i++ {
			innerUserId := set.ConvertUserID(users[i])
			innerItemId := set.ConvertItemID(items[i])
			set.itemRatings[innerItemId][innerUserId] = ratings[i]
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
	begin, end := 0, 0 // todo 数据集划分
	for i := 0; i < k; i++ {
		end += foldSize
		if i < set.Length()%k {
			end++
		}
		// Test set
		testIndex := perm[begin:end]
		testFolds[i] = NewTrainSet(selectInt(set.interactionUsers, testIndex),
			selectInt(set.interactionItems, testIndex),
			selectFloat(set.interactionRatings, testIndex),
		)
		// Train set
		trainIndex := concatenate(perm[0:begin], perm[end:set.Length()])
		trainFolds[i] = NewTrainSet(selectInt(set.interactionUsers, trainIndex),
			selectInt(set.interactionItems, trainIndex),
			selectFloat(set.interactionRatings, trainIndex),
		)
		begin = end
	}
	return trainFolds, testFolds
}

func (set *TrainSet) Split(testSize int) (TrainSet, TrainSet) {
	return TrainSet{}, TrainSet{}
}
func NewTrainSet(users, items []int, ratings []float64) TrainSet {
	set := TrainSet{}
	set.interactionUsers = users
	set.interactionItems = items
	set.interactionRatings = ratings

	// 创建userID -> innerUserID的映射
	set.innerUserIDs = make(map[int]int)
	for _, userID := range set.interactionUsers {
		if _, exist := set.innerUserIDs[userID]; !exist {
			set.innerUserIDs[userID] = set.userCount
			set.userCount++
		}
	}
	// 创建itemID -> innerUserID的映射
	set.innerItemIDs = make(map[int]int)
	for _, itemID := range set.interactionItems {
		if _, exist := set.innerItemIDs[itemID]; !exist {
			set.innerItemIDs[itemID] = set.itemCount
			set.itemCount++
		}
	}

	return set
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

	interactionUsers := make([]int, 0)
	interactionItems := make([]int, 0)
	interactionRatings := make([]float64, 0)

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
		interactionUsers = append(interactionUsers, user)
		interactionItems = append(interactionItems, item)
		interactionRatings = append(interactionRatings, float64(ratings))
	}
	return NewTrainSet(interactionUsers, interactionItems, interactionRatings)
}
