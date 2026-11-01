import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Performance } from './Performance'
import { Backtests } from './Backtests'
import { performance, backtests } from '../mocks/data'

const hooks=vi.hoisted(()=>({performanceStatus:'ok' as 'ok'|'error',reload:vi.fn(),list:backtestsPlaceholder()}))
function backtestsPlaceholder(){return [] as {id:string;label:string;hypothesis_name:string|null;start:string;end:string;total_return:number}[]}
vi.mock('../api/hooks',()=>({useBacktestPerformance:()=>hooks.performanceStatus==='error'?{status:'error',reload:hooks.reload}:{status:'ok',data:performance,reload:hooks.reload},useBacktests:()=>({status:'ok',data:hooks.list,reload:vi.fn()})}))

describe('flujos de rendimiento',()=>{
 afterEach(()=>{cleanup();hooks.performanceStatus='ok';hooks.reload.mockClear();hooks.list=backtests})
 it('ErrorState permite reintentar la carga fallida',()=>{hooks.performanceStatus='error';render(<Performance id="41" navigate={()=>{}}/>);fireEvent.click(screen.getByRole('button',{name:'Reintentar'}));expect(hooks.reload).toHaveBeenCalledOnce()})
 it('al seleccionar dos backtests abre la comparación con el segundo en vs',()=>{const navigate=vi.fn();render(<Backtests navigate={navigate}/>);fireEvent.click(screen.getByRole('checkbox',{name:/Rotación sectorial/}));fireEvent.click(screen.getByRole('checkbox',{name:/Reversión a la media/}));fireEvent.click(screen.getByRole('button',{name:/Comparar 2/}));expect(navigate).toHaveBeenCalledWith('/rendimiento/backtest/41?vs=backtest:42')})
})
