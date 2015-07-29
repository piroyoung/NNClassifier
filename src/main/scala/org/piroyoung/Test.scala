package org.piroyoung

import org.apache.spark.{SparkConf, SparkContext}
import org.piroyoung.classficate.FeedForwardNetwork
import org.piroyoung.linalg.ColVector
import java.io.PrintWriter
import scala.io.Source

import scala.io.Source

/**
 * Created by piroyoung on 7/25/15.
 */
object Test {
  def main(args: Array[String]) {

    val ff = FeedForwardNetwork(784, 32, 16, 10).setEta(1)

//    val conf = new SparkConf().setMaster("local[4]").setAppName("myJob")
//    val sc = new SparkContext(conf)
//
//    val dat = sc.textFile("src/main/resources/train.csv").map(_.stripMargin)
    val dat = Source.fromFile("src/main/resources/train.csv").getLines().map(_.stripMargin).toSeq

    val input = dat.zipWithIndex
      .filter(x => x._2 > 0)
      .map(_._1.split(',').map(_.toDouble))
      .map(x => (ColVector(x.drop(1).toSeq) / 255 , x(0)))

//    input.foreach(x => println(x._2))
//
    ff.fit(input, 10, 10)
    ff.saveAsTextFile("src/main/resources/out/model32-16.ffn")
//
////    println(ff.toString())
//    ff.load("src/main/resources/out/model.ffn")
//    for(d <- input) {
//      val v = ff.predict(d._1)
//      val label = d._2
//
//      println(v.toSeq.indexOf(v.toSeq.max) == label)
//
//    }





  }
}
