package BIDMach.allreduce.binder

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}
import BIDMach.models.Model
import BIDMat.{FMat, IMat, Mat}


/**
  * Linearize input model mats, and elastic-average update to the same model.
  *
  * @param model
  * @param alpha
  */
class ElasticAverageBinder(model: Model, alpha: Double) extends AllreduceBinder {

  override lazy val totalDataSize: Int = {
    var ret = 0
    model.modelmats.synchronized {
      for (mat <- model.modelmats) {
        ret += mat.length
      }
    }
    ret
  }

  lazy val tmpUpdateMats: Array[FMat] = model.modelmats.map(m => FMat.make(m.dims))

  lazy val countMats: Array[IMat] = model.modelmats.map(m => IMat.make(m.dims))

  lazy val diffMats: Array[Mat] = model.modelmats.map(m => IMat.make(m.dims))


  override def dataSource: DataSource = inputRequest => {

    println(s"--Dumping model data at ${inputRequest.iteration}--")
    val ret: Array[Float] = new Array[Float](totalDataSize)

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize
    var i = model.modelmats.length - 1

    while (i >= 0) {
      val mat = model.modelmats(i)
      mat.synchronized {
        val contentData = FMat(mat).contents.data
        current -= contentData.length
        System.arraycopy(contentData, 0, ret, current, contentData.length)
      }
      i -= 1
    }

    AllReduceInput(ret)

  }

  private def averageValueOrElse(sum: Float, count: Int): Option[Float] = {
    count match {
      case 0 => Option.empty
      case _ => Some(sum / count)
    }
  }

  override def dataSink: DataSink = reducedOutput => {
    println(s"-- Averaging model of iteration ${reducedOutput.iteration}--")

    val data = reducedOutput.data
    val count = reducedOutput.count

    assert(data.length == totalDataSize, "Reduced output should be the same as as model")

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize
    var i = model.modelmats.length - 1

    // trasnfer to fmat so that we can just add average



    while (i >= 0) {
      val mat = model.modelmats(i)
      val contentData = FMat(mat).contents.data
      val contentLength = contentData.length

      val tmpMat = tmpUpdateMats(i)
      val countMat = countMats(i)

      current -= contentData.length
      System.arraycopy(data, current, tmpMat.data, 0, contentLength)
      System.arraycopy(count, current, countMat.data, 0, contentLength)

      tmpMat ~ tmpMat / countMat
      diffMats(i) = tmpMat - mat
      diffMats(i) ~ diffMats(i) *@ alpha
      mat ~ mat + diffMats(i)
      i -= 1
    }

    assert(current == 0, "current should be zero after iteration")

  }

}

