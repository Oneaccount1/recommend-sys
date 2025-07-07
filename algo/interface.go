package algo

import "recommend-sys/data"

type Algorithm interface {
	Predict(userID, itemID int) float64
	Fit(trainData data.Set)
}
