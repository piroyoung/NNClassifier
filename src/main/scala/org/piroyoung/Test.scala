package org.piroyoung

import org.piroyoung.classficate.FeedForwardNetwork
import org.piroyoung.linalg.ColVector

import scala.io.Source

/**
 * Created by piroyoung on 7/25/15.
 */
object Test {
  def main(args: Array[String]) {

    //learn()
    //verily()

  }

  def learn(): Unit = {
    val ff = FeedForwardNetwork(784, 32, 16, 10).setEta(1)
    val dat = Source.fromFile("src/main/resources/train.csv").getLines().map(_.stripMargin).toSeq
    val input = dat.zipWithIndex
      .filter(x => x._2 > 0)
      .map(_._1.split(',').map(_.toDouble))
      .map(x => (ColVector(x.drop(1).toSeq) / 255, x(0)))

    ff.fit(input, 10, 10)
    ff.saveAsTextFile("src/main/resources/out/model32-16.ffn")
  }

  def verify: Unit = {

    val ff = FeedForwardNetwork.load("src/main/resources/out/model32-16_82.1.ffn")
    val dat = Source.fromFile("src/main/resources/train.csv").getLines().map(_.stripMargin).toSeq

    val input = dat.zipWithIndex
      .filter(x => x._2 > 0)
      .map(_._1.split(',').map(_.toDouble))
      .map(x => (ColVector(x.drop(1).toSeq) / 255, x(0)))

    val result = input
      .map(x => (ff.predictLabel(x._1), x._2))
      .map(x => (if (x._1 == x._2) 1.0 else 0.0, 1.0))
      .reduce((x, y) => (x._1 + y._1, x._2 + y._2))

    println(result._1 / result._2)

  }
}
