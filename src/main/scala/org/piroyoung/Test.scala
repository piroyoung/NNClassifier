package org.piroyoung

import org.piroyoung.classficate.FeedForwardNetwork
import org.piroyoung.linalg.ColVector

import scala.io.Source

/**
 * Created by piroyoung on 7/25/15.
 */
object Test {
  def main(args: Array[String]) {

    val ff = FeedForwardNetwork(784, 128, 64, 32, 10)

//    val conf = new SparkConf().setMaster("local[4]").setAppName("myJob")
//    val sc = new SparkContext(conf)

//    val dat = sc.textFile("src/main/resources/train.csv")
    val dat = Source.fromFile("src/main/resources/train.csv").getLines()

    val input = dat.zipWithIndex
      .filter(x => x._2 > 0)
      .map(_._1.split(',').map(_.toDouble))
      .map(x => (ColVector(x.drop(1).toSeq), x(0))).toSeq


    ff.fit(input ,10, iter = 1)

    input.map(_._1).map(ff.predict(_).t.toString + "\n")

    ff.predict(ColVector(Seq(1,0,1))).t.toString.foreach(println)




  }
}
