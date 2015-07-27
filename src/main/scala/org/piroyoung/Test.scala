package org.piroyoung

import org.piroyoung.classficate.FeedForwardNetwork
import org.piroyoung.linalg.ColVector

/**
 * Created by piroyoung on 7/25/15.
 */
object Test {
  def main(args: Array[String]) {

    val ff = new FeedForwardNetwork(3, 2, 2, 2, 3)

    val input = ColVector(Seq(1,0,0))
    val answer = ColVector(Seq(1,0,0))

    val input2 = ColVector(Seq(1,0,1))
    val answer2 = ColVector(Seq(0,1,0))

    for(i <- Range(0,800)) {
      println("---")
      println("signal1:" + ff.fit(input, answer).t.toString)
      println("signal2:" + ff.fit(input2, answer2).t.toString)
    }

  }
}
