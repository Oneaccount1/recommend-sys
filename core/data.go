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

type DataSet struct {
	Ratings []float64
	Users   []int
	Items   []int
}

func NewRawSet(users, items []int, ratings []float64) DataSet {
	return DataSet{
		Users:   users,
		Items:   items,
		Ratings: ratings,
	}
}

func (d *DataSet) Length() int {
	return len(d.Ratings)
}
func (d *DataSet) Index(i int) (int, int, float64) {
	return d.Users[i], d.Items[i], d.Ratings[i]
}

func (d *DataSet) SubSet(indices []int) DataSet {
	return NewRawSet(selectInt(d.Users, indices),
		selectInt(d.Items, indices),
		selectFloat(d.Ratings, indices),
	)
}

func (d *DataSet) KFold(k int, seed int64) ([]TrainSet, []DataSet) {
	trainFolds := make([]TrainSet, k)
	testFolds := make([]DataSet, k)
	rand.New(rand.NewSource(seed))
	perm := rand.Perm(d.Length())
	foldSize := d.Length() / k
	begin, end := 0, 0
	for i := 0; i < k; i++ {
		end += foldSize
		if i < d.Length()%k {
			end++
		}

		// Test trainSet
		testIndex := perm[begin:end]
		testFolds[i] = d.SubSet(testIndex)
		// Train trainSet
		trainIndex := concatenate(perm[0:begin], perm[end:d.Length()])
		trainFolds[i] = NewTrainSet(d.SubSet(trainIndex))
		begin = end
	}
	return trainFolds, testFolds
}

type Set map[int]interface{}

// TrainSet 由原来的Map实现 Map[int]map[int]float   用户-物品-评分
// 转换为 map[int][int] 定位矩阵的行列位置   [][]float 评分矩阵
// 加速算法运行
type TrainSet struct {
	DataSet
	userCount int
	itemCount int

	innerUserIDs map[int]int
	innerItemIDs map[int]int

	userRatings [][]float64
	itemRatings [][]float64
}

const newID = -1

// UserSet   获取数据集所有User
func (set *TrainSet) UserSet() Set {
	return unique(set.Users)
}

// ItemSet   获取所有数据集Item
func (set *TrainSet) ItemSet() Set {
	return unique(set.Items)
}

// RatingRange 获取数据集评分的范围
func (set *TrainSet) RatingRange() (float64, float64) {
	return floats.Min(set.Ratings), floats.Max(set.Ratings)
}

// GlobalMean 获取数据集中Rating的平均值
func (set *TrainSet) GlobalMean() float64 {
	return stat.Mean(set.Ratings, nil)
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

		for i := 0; i < len(set.Users); i++ {
			innerUserID := set.ConvertUserID(set.Users[i])
			innerItemID := set.ConvertItemID(set.Items[i])
			set.userRatings[innerUserID][innerItemID] = set.Ratings[i]
		}
	}
	return set.userRatings
}

func (set *TrainSet) ItemRatings() [][]float64 {
	if set.itemRatings == nil {
		set.itemRatings = newNanMatrix(set.itemCount, set.userCount)
		for i := 0; i < len(set.Users); i++ {
			innerUserId := set.ConvertUserID(set.Users[i])
			innerItemId := set.ConvertItemID(set.Items[i])
			set.itemRatings[innerItemId][innerUserId] = set.Ratings[i]
		}
	}
	return set.itemRatings
}

func (set *TrainSet) Split(testSize int) (TrainSet, TrainSet) {
	return TrainSet{}, TrainSet{}
}
func NewTrainSet(rowSet DataSet) TrainSet {
	set := TrainSet{}
	set.DataSet = rowSet

	// 创建userID -> innerUserID的映射
	set.innerUserIDs = make(map[int]int)
	for _, userID := range set.Users {
		if _, exist := set.innerUserIDs[userID]; !exist {
			set.innerUserIDs[userID] = set.userCount
			set.userCount++
		}
	}
	// 创建itemID -> innerUserID的映射
	set.innerItemIDs = make(map[int]int)
	for _, itemID := range set.Items {
		if _, exist := set.innerItemIDs[itemID]; !exist {
			set.innerItemIDs[itemID] = set.itemCount
			set.itemCount++
		}
	}

	return set
}

// LoadDataFromBuiltIn Load build in data set
func LoadDataFromBuiltIn(dataSetName string) DataSet {
	// Extract data set information
	dataSet, exist := buildInDataSets[dataSetName]
	if !exist {
		log.Fatal("no such data set", dataSetName)
	}
	const dataFolder = "data"
	const tempFolder = "temp"
	dataFileName := filepath.Join(dataFolder, dataSet.path)
	if _, err := os.Stat(dataFileName); os.IsNotExist(err) {
		zipFileName, _ := downloadFromUrl(dataSet.url, tempFolder)
		unzip(zipFileName, dataFolder)
	}
	return LoadDataFromFile(dataFileName, dataSet.sep)
}

// LoadDataFromFile Load data from file
func LoadDataFromFile(fileName string, sep string) DataSet {
	users := make([]int, 0)
	items := make([]int, 0)
	ratings := make([]float64, 0)
	// Open file
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	// Read CSV file
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, sep)
		user, _ := strconv.Atoi(fields[0])
		item, _ := strconv.Atoi(fields[1])
		rating, _ := strconv.Atoi(fields[2])
		users = append(users, user)
		items = append(items, item)
		ratings = append(ratings, float64(rating))
	}
	return NewRawSet(users, items, ratings)
}
