/* generic */
const mapSel = (sel, fn) => {
  document.querySelectorAll(sel).forEach(fn)
}
const addClass = (sel, className) => {
  mapSel(sel, (ele) => {
    ele.classList.add(className)
  })
}
const removeClass = (sel, className) => {
  mapSel(sel, (ele) => {
    ele.classList.remove(className)
  })
}

/* show/hide */
const hiddenClass = 'hidden'
const show = (sel) => {
  removeClass(sel, hiddenClass)
}
const hide = (sel) => {
  addClass(sel, hiddenClass)
}

/* page */
const eles = [ 'done', 'running', 'up-next' ]
const state = {}
eles.forEach(
  (parentId) => {
    document.getElementById(parentId).addEventListener(
      'click',
      () => {
        let expandState = state[parentId]
        expandState = expandState === undefined || !expandState
        state[parentId] = expandState
        const displayFunc = expandState ? hide : show
        displayFunc(`.child-${parentId}`)
      }
    )
  }
)
