/*
Go was designed with concurrency in mind

Go attempts to not share the data across threads, instead
they share a communication channel. In order to communicate
across the threads

Don't share memory, share the communication about the data

Goroutines: Lightweight thread; keyword go

A basic go routine can be created by putting the 'go' keyword
in front of the relevant function call.

However, may need to introduce a sleep, as the new thread is non-blocking
meaning things may be in the wrong order

Buffered channels are slightly different. Basically, they take multipple
values into the channel.

To do this declare:

done := make(chan bool, 2)
This allows two items to be pushed onto the channel.
This effectively means how many items can be pushed onto the channel
before it can be read.

So I can't put a second item on until first one is read in unbuffered
channel (reading is done via: <- done)

i.e. how many times can i write before that data has to be read ?


Select statements are like switch but for channels
it will execute statements that are "ready". If more than one is ready
it will choose which of those that are ready to execute at random
If none are ready, it will block unless default is defined


*/

package main

import (
	"sort"
)

type Record map[string]interface{}
type Records []Record

type DirectionGroup map[string]Records
type CountryGroup map[string]DirectionGroup
type IdGroup map[string]CountryGroup

func (grp *IdGroup) appendRecord(record Record) {
	grouped := *grp
	if grouped[record["Ats_ident"].(string)] == nil {
		country := make(CountryGroup)
		direction := make(DirectionGroup)

		direction[record["Direction"].(string)] = append(grouped[record["Ats_ident"].(string)][record["Ctry"].(string)][record["Direction"].(string)], record)
		country[record["Ctry"].(string)] = direction
		grouped[record["Ats_ident"].(string)] = country
	} else if grouped[record["Ats_ident"].(string)][record["Ctry"].(string)] == nil {
		country := grouped[record["Ats_ident"].(string)]
		direction := make(DirectionGroup)

		direction[record["Direction"].(string)] = append(grouped[record["Ats_ident"].(string)][record["Ctry"].(string)][record["Direction"].(string)], record)
		country[record["Ctry"].(string)] = direction
		grouped[record["Ats_ident"].(string)] = country
	} else {
		grouped[record["Ats_ident"].(string)][record["Ctry"].(string)][record["Direction"].(string)] = append(grouped[record["Ats_ident"].(string)][record["Ctry"].(string)][record["Direction"].(string)], record)
	}
}

func main() {
	grouped := make(IdGroup)

	r := make(Record)
	r["Foo"] = "Foo"
	r["Bar"] = "Bar"
	r["Baz"] = "Baz"
	r["Bax"] = "Bax"
	r["Cax"] = "Cax"
	r["Ctry"] = "US"
	r["Direction"] = "E"
	r["Number"] = 1
	r["Ats_ident"] = "US435"

	x := make(Record)
	x["Foo"] = "Foo"
	x["Bar"] = "Bar"
	x["Baz"] = "Baz"
	x["Bax"] = "Bax"
	x["Cax"] = "Cax"
	x["Ctry"] = "US"
	x["Direction"] = "E"
	x["Number"] = 2
	x["Ats_ident"] = "US435"

	y := make(Record)
	y["Foo"] = "Foo"
	y["Bar"] = "Bar"
	y["Baz"] = "Baz"
	y["Bax"] = "Bax"
	y["Cax"] = "Cax"
	y["Ctry"] = "US"
	y["Direction"] = "E"
	y["Number"] = 3
	y["Ats_ident"] = "US435"

	s := make(Record)
	s["Foo"] = "Foo"
	s["Bar"] = "Bar"
	s["Baz"] = "Baz"
	s["Bax"] = "Bax"
	s["Cax"] = "Cax"
	s["Ctry"] = "US"
	s["Direction"] = "W"
	s["Number"] = 1
	s["Ats_ident"] = "US435"

	t := make(Record)
	t["Foo"] = "Foo"
	t["Bar"] = "Bar"
	t["Baz"] = "Baz"
	t["Bax"] = "Bax"
	t["Cax"] = "Cax"
	t["Ctry"] = "UK"
	t["Direction"] = "W"
	t["Number"] = 1
	t["Ats_ident"] = "US435"

	u := make(Record)
	u["Foo"] = "Foo"
	u["Bar"] = "Bar"
	u["Baz"] = "Baz"
	u["Bax"] = "Bax"
	u["Cax"] = "Cax"
	u["Ctry"] = "UK"
	u["Direction"] = "W"
	u["Number"] = 1
	u["Ats_ident"] = "US999"

	grouped.appendRecord(r)
	grouped.appendRecord(s)
	grouped.appendRecord(t)
	grouped.appendRecord(u)
	grouped.appendRecord(y)
	grouped.appendRecord(x)

	uniqueATSVals := [2]string{"US435", "US999"}
	uniqueCtryVals := [2]string{"US", "UK"}
	uniqueDirections := [2]string{"E", "W"}

	for _, atsIdent := range uniqueATSVals {
		for _, country := range uniqueCtryVals {
			for _, direction := range uniqueDirections {
				records := grouped[atsIdent][country][direction]
				sort.Slice(records[:], func(i, j int) bool {
					return records[i]["Number"].(int) < records[j]["Number"].(int)
				})
				//fmt.Println(atsIdent, country, direction, records)
			}
		}
	}

}
