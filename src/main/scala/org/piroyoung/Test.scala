package org.piroyoung

import org.piroyoung.classficate.FeedForwardNetwork
import org.piroyoung.linalg.ColVector
import org.piroyoung.linalg.Functions._

import scala.io.Source

/**
 * Created by piroyoung on 7/25/15.
 */
object Test {
  def main(args: Array[String]) {

    learn()
//    verify()

  }

  def learn(): Unit = {
//    val ff = FeedForwardNetwork(784, 64, 32, 10).setEta(0.1).setActivator(dropSigmoid).setBatchSize(64)
    val ff = FeedForwardNetwork.load("src/main/resources/out/model_64_32_5.ffn").setEta(0.1).setActivator(dropSigmoid).setBatchSize(64)
    val dat = Source.fromFile("src/main/resources/train.csv").getLines().map(_.stripMargin).toSeq

    val input = dat.zipWithIndex
      .filter(x => x._2 > 0)
      .map(_._1.split(',').map(_.toDouble))
      .map(x => (ColVector(x.drop(1).toSeq) / 255, x(0)))

    ff.fit(input, 5)
    ff.saveAsTextFile("src/main/resources/out/model_64_32_10.ffn")
  }

  def verify(): Unit = {

    val ff = FeedForwardNetwork.load("src/main/resources/out/model_64_32_5.ffn")
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
